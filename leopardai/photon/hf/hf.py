import base64
from functools import cached_property
from io import BytesIO
import os
import re
import tempfile
from typing import List, Union, Optional, Dict, Any

from huggingface_hub import model_info, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from loguru import logger

from leopardai.registry import Registry
from leopardai.photon.base import schema_registry
from leopardai.photon import Photon, PNGResponse, FileParam, HTTPException
from .hf_utils import (
    pipeline_registry,
    img_param_to_img,
    hf_missing_package_error_message,
    hf_try_explain_run_exception,
)
from .hf_dependencies import hf_pipeline_dependencies

task_cls_registry = Registry()


HF_DEFINED_TASKS = [
    # diffusers
    "text-to-image",
    "image-to-image",
    # transformers
    "audio-classification",
    "automatic-speech-recognition",
    "conversational",
    "depth-estimation",
    "document-question-answering",
    "feature-extraction",
    "fill-mask",
    "image-classification",
    "image-to-text",
    "object-detection",
    "question-answering",
    "sentiment-analysis",
    "summarization",
    "table-question-answering",
    "text-classification",
    "text-generation",
    "text2text-generation",
    "token-classification",
    "translation",
    "video-classification",
    "visual-question-answering",
    "vqa",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
    # sentence-transformers
    "sentence-similarity",
]

# This is a manually maintained list of mappings from model name to
# tasks. Somehow these models are not properly annotated in the Huggingface
# Hub.
_MANUALLY_ANNOTATED_MODEL_TO_TASK = {
    "hf-internal-testing/tiny-stable-diffusion-torch": "text-to-image",
}

HUGGING_FACE_SCHEMAS = ["hf", "huggingface"]


def _get_transformers_base_types():
    import transformers

    return (transformers.PreTrainedModel, transformers.Pipeline)


def is_transformers_model(model):
    return isinstance(model, _get_transformers_base_types())


class HuggingfacePhoton(Photon):
    photon_type: str = "hf"
    hf_task: str = "undefined (please override this in your derived class)"
    requirement_dependency: Optional[List[str]] = []

    @property
    def _requirement_dependency(self) -> List[str]:
        """
        Returns the list of requirements that are needed to run the photon.

        This method also checks huggingface hub for any additional dependencies
        that is specified in the "requirements.txt" file of the model, to conform
        with the best practices of the huggingface ecosystem.
        """
        # First, get the overridden base class requirement_dependency
        deps: List[str] = super()._requirement_dependency
        # Add the dependencies from the model's requirements.txt file
        try:
            logger.trace(
                "Trying to download the requirements.txt file for"
                f" {self.hf_model} {self.hf_revision}."
            )
            model_deps_file = hf_hub_download(
                repo_id=self.hf_model,
                filename="requirements.txt",
                revision=self.hf_revision,
            )
            with open(model_deps_file, "r") as f:
                model_deps = f.readlines()
                deps.extend([d.strip() for d in model_deps])
                logger.trace(
                    "Adding dependencies from the model's requirements.txt file:"
                    f" {model_deps}."
                )
        except EntryNotFoundError:
            # If the model does not have a requirements.txt file, we can safely
            # ignore it.
            logger.trace("Model does not have a requirements.txt file.")
            pass
        except Exception as e:
            # If any other error occurs, we issue a warning, but don't really
            # exit loud, to make sure that the users have a good develop experience.
            logger.warning(
                "An error occurred while trying to download the requirements.txt"
                " file from the model's repository. We will ignore this error and"
                " continue with the default dependencies, but please be noted that"
                " some of the required dependencies might not be installed. Error"
                f" details: {e}"
            )
        # Add manually maintained dependencies to the list.
        if self.hf_model in hf_pipeline_dependencies:
            pipeline_specific_deps = hf_pipeline_dependencies[self.hf_model]
            for d in pipeline_specific_deps:
                if d not in deps:
                    deps.append(d)
        return deps

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.hf_task not in HF_DEFINED_TASKS:
            raise ValueError(
                f"You made a programming error: the task {cls.hf_task} is not a"
                " supported task defined in HuggingFace. If you believe this is an"
                " error, please file an issue."
            )
        task_cls_registry.register(cls.hf_task, cls)

    @classmethod
    def supported_tasks(cls):
        """
        Returns the set of supported tasks.
        """
        return task_cls_registry.keys()

    @classmethod
    def _parse_model_str(cls, model_str):
        model_parts = model_str.split(":")
        if len(model_parts) not in (2, 3):
            raise ValueError(
                f'Unsupported Huggingface model: "{model_str}" (can not parse model'
                " name). Huggingface model spec should be in the form of"
                " hf:<model_name>[@<revision>], or"
                " hf:<task_name>:<model_name>[@<revision>]."
            )
        schema = model_parts[0]
        if schema not in HUGGING_FACE_SCHEMAS:
            # In theory, this should not happen - the schema should be
            # automatically checked by the photon registry, but we'll check it
            # here just in case.
            raise ValueError(
                f'Unsupported Huggingface model: "{model_str}" (unknown schema:'
                f' "{schema}")'
            )
        hf_model_id = model_parts[-1]
        if "@" in hf_model_id:
            hf_model_id, revision = hf_model_id.split("@")
        else:
            revision = None
        mi = model_info(hf_model_id, revision=revision)

        if len(model_parts) == 3:
            hf_task = model_parts[1]
            logger.info(
                f"You have explicitly specified the hugging face task as {hf_task}."
            )
            if hf_task != mi.pipeline_tag:
                logger.info(
                    f"Specified task {hf_task} does not match the task of the model"
                    f" pipeline tag {mi.pipeline_tag}. Kindly make sure that you have"
                    " specified the correct task that this model supports."
                )
        elif hf_model_id in _MANUALLY_ANNOTATED_MODEL_TO_TASK:
            hf_task = _MANUALLY_ANNOTATED_MODEL_TO_TASK[hf_model_id]
        else:
            hf_task = mi.pipeline_tag
            if hf_task is None:
                raise AttributeError(
                    f'Unsupported Huggingface model: "{model_str}" (the model did not'
                    " specify a task).\nUsually, this means that the model creator did"
                    " not intend to publish the model as a pipeline, and is only using"
                    " HuggingFace Hub as a storage for the model weights and misc"
                    " files. Thus, it is not possible to run the model automatically."
                    " This is not a bug of leopard AI library, but rather a limitation"
                    " of the hf ecosystem.\n\nAs a possible solution, you can try to"
                    " access the corresponding model page at"
                    f" https://huggingface.co/{hf_model_id} and see if the model"
                    " creator provided any instructions on how to run the model. You"
                    " can then wrap this model into a custom Photon with relatively"
                    " easy scaffolding. Check out the documentation at"
                    " https://www.leopard.ai/docs/walkthrough/anatomy_of_a_photon for"
                    " more details."
                )

        if hf_task not in HF_DEFINED_TASKS:
            raise ValueError(
                f'Unsupported Huggingface model: "{model_str}" (task: {hf_task}). This'
                " task is not supported by the library yet. If you would like us to"
                " add support for this task type, please let us know by opening an"
                " issue at https://github.com/leopard/leopardai/issues/new/choose."
                f"\nCurrently supported HF tasks are: {cls.supported_tasks()}."
            )

        # 8 chars should be enough to identify a commit
        hf_revision = mi.sha[:8]
        if len(model_parts) == 3:
            model = f"{schema}:{hf_task}:{hf_model_id}@{hf_revision}"
        else:
            model = f"{schema}:{hf_model_id}@{hf_revision}"

        return model, hf_task, hf_model_id, hf_revision

    def __init__(self, name: str, model: str):
        model, hf_task, hf_model_id, hf_revision = self._parse_model_str(model)
        super().__init__(name, model)

        self.hf_model = hf_model_id
        self.hf_revision = hf_revision
        self.hf_task = hf_task

    @property
    def metadata(self):
        res = super().metadata
        res.pop("py_obj")
        res.update({
            "task": self.hf_task,
        })
        return res

    @cached_property
    def pipeline(self):
        pipeline_creator = pipeline_registry.get(self.hf_task)
        if pipeline_creator is None:
            raise ValueError(f"Could not find pipeline creator for {self.hf_task}")
        logger.info(
            f"Creating pipeline for {self.hf_task}(model={self.hf_model},"
            f" revision={self.hf_revision}).\n"
            "HuggingFace download might take a while, please be patient..."
        )
        logger.info(
            "Note: HuggingFace caches the downloaded models in ~/.cache/huggingface/"
            " (or C:\\Users\\<username>\\.cache\\huggingface\\ on Windows). If you"
            " have already downloaded the model before, the download should be much"
            " faster. If you run out of disk space, you can delete the cache folder."
        )
        try:
            pipeline = pipeline_creator(
                task=self.hf_task,
                model=self.hf_model,
                revision=self.hf_revision,
            )
        except ImportError as e:
            # Huggingface has a mechanism that detects dependencies, and then prints dependent
            # libraries in the error message. When this happens, we want to parse the error and
            # then tell the user what they should do.
            # See https://github.com/huggingface/transformers/blob/ce2e7ef3d96afaf592faf3337b7dd997c7ad4928/src/transformers/dynamic_module_utils.py#L178
            # for the source code that prints the error message.
            pattern = (
                "This modeling file requires the following packages that were not found"
                " in your environment: (.*?). Run `pip install"
            )
            match = re.search(pattern, e.msg)
            if match:
                missing_packages = match.group(1).split(", ")
                raise ImportError(
                    hf_missing_package_error_message(self.hf_task, missing_packages)
                ) from e
            else:
                raise e

        return pipeline

    def init(self):
        super().init()
        # access pipeline here to trigger download and load
        self.pipeline

    def _run_pipeline(self, *args, **kwargs):
        import torch

        # autocast causes invalid value (and generates black images) for text-to-image and image-to-image
        no_auto_cast_set = ("text-to-image", "image-to-image")
        if torch.cuda.is_available() and self.hf_task not in no_auto_cast_set:
            with torch.autocast(device_type="cuda"):
                return self.pipeline(*args, **kwargs)
        else:
            return self.pipeline(*args, **kwargs)

    @classmethod
    def create_from_model_str(cls, name, model_str):
        _, hf_task, _, _ = cls._parse_model_str(model_str)
        task_cls = task_cls_registry.get(hf_task)
        if task_cls is None:
            raise ValueError(
                f"leopard currently does not support the specified task: {hf_task}. If"
                " you would like us to support this task, please let us know by"
                " opening an issue"
                " at https://github.com/leopardai/leopardai/issues/new/choose, and"
                " kindly include the specific model that you are trying to run for"
                " debugging purposes: {model_str}"
            )
        return task_cls(name, model_str)

    @classmethod
    def load(cls, photon_file, metadata):
        name = metadata["name"]
        model = metadata["model"]
        return cls.create_from_model_str(name, model)


def _get_generated_text(res):
    if isinstance(res, str):
        return res
    elif isinstance(res, dict):
        return res["generated_text"]
    elif isinstance(res, list):
        if len(res) == 1:
            return _get_generated_text(res[0])
        else:
            return [_get_generated_text(r) for r in res]
    else:
        raise ValueError(f"Unsupported result type in _get_generated_text: {type(res)}")


class HuggingfaceTextGenerationPhoton(HuggingfacePhoton):
    hf_task: str = "text-generation"
    requirement_dependency: Optional[List[str]] = ["ctransformers"]

    @Photon.handler(
        "run",
        example={
            "inputs": "I enjoy walking with my cute dog",
            "max_new_tokens": 50,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
        },
    )
    def run(
        self,
        inputs: Union[str, List[str]],
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = 1.0,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        max_time: Optional[float] = None,
        return_full_text: bool = True,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        try:
            res = self._run_pipeline(
                inputs,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                max_time=max_time,
                return_full_text=return_full_text,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                **kwargs,
            )
            return _get_generated_text(res)
        except Exception as e:
            raise hf_try_explain_run_exception(e)

    def answer(self, question, history):
        history.append({"role": "user", "content": question})

        # TODO: should limit the number of history messages to include into the
        # prompt so that it doesn't exceed the max context length of the model
        history_prompt = os.linesep.join(
            [f"{h['role']}: {h['content']}" for h in history]
        )
        prompt = f"""\
The following is a friendly conversation between a user and an assistant. The assistant is talkative and provides lots of specific details from its context. If the assistant does not know the answer to a question, it truthfully says it does not know.
Current conversation:
{history_prompt}

assistant:
"""
        response = self._run_pipeline(prompt, return_full_text=False)[0][
            "generated_text"
        ]
        history.append({"role": "assistant", "content": response})
        messages = [
            (history[i]["content"], history[i + 1]["content"])
            for i in range(0, len(history) - 1, 2)
        ]
        return messages, history

    @Photon.handler(mount=True)
    def ui(self):
        import gradio as gr

        blocks = gr.Blocks()
        with blocks:
            chatbot = gr.Chatbot(label=f"Chatbot ({self.hf_model})")
            state = gr.State([])
            with gr.Row():
                txt = gr.Textbox(
                    show_label=False, placeholder="Enter text and press enter"
                )
            txt.submit(self.answer, [txt, state], [chatbot, state])
        return blocks