import uuid
import logging
from typing import Any, Dict, Optional

import dspy
from langfuse import Langfuse

from dspy.utils import BaseCallback
from dspy.utils.callback import ACTIVE_CALL_ID

logger = logging.getLogger(__name__)

class LangfuseCallback(BaseCallback):
    def __init__(self, public_key: Optional[str] = None, secret_key: Optional[str] = None, host: Optional[str] = None):
        # Initialize the Langfuse client. Credentials can also be set via environment variables.
        self.langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host="https://cloud.langfuse.com"
        )
        # Validate credentials
        try:
            self.langfuse.auth_check()
        except Exception as e:
            logger.warning(f"Langfuse auth check failed: {e}")

        # Dictionary to store references to Langfuse trace/generation objects by call_id
        self.calls = {}

    def on_module_start(self, call_id: str, instance: Any, inputs: Dict[str, Any]):
        # Create a new trace for this module call.
        # You can customize `name`, `metadata`, `input` as needed.
        try:
            trace = self.langfuse.trace(
                name=f"Module-{instance.__class__.__name__}",
                input=inputs,
                metadata={"module_class": instance.__class__.__name__},
                tags=["module_call"]
            )
            self.calls[call_id] = trace
        except Exception as e:
            logger.warning(f"Failed to create trace for module start: {e}")

    def on_module_end(
        self,
        call_id: str,
        outputs: Optional[Any],
        exception: Optional[Exception] = None,
    ):
        # Complete or update the trace associated with this module call.
        trace = self.calls.get(call_id)
        if trace is None:
            return

        try:
            if exception:
                # If an error occurred, mark the trace with an error level and status message.
                trace.update(
                    output=outputs if outputs is not None else {},
                    level="ERROR",
                    status_message=str(exception)
                )
            else:
                # If successful, update the trace with the outputs.
                trace.update(
                    output=outputs if outputs is not None else {}
                )

        except Exception as e:
            logger.warning(f"Failed to update/end trace for module end: {e}")

        # Clean up reference
        self.calls.pop(call_id, None)

    def on_lm_start(self, call_id: str, instance: Any, inputs: Dict[str, Any]):
        # For LM calls, we create a generation nested under the current active trace.
        # We rely on ACTIVE_CALL_ID to get the parent call, but here we assume LM calls happen
        # within a module's trace. If needed, adjust logic to find the correct parent trace.
        parent_call_id = ACTIVE_CALL_ID.get()
        if parent_call_id is None:
            # If there's no parent trace, we can either create one or just log a warning.
            logger.warning("No parent trace found for LM call. Creating a standalone trace.")
            trace = self.langfuse.trace(name="Standalone-LM-Call", input=inputs, tags=["lm_call"])
            self.calls[call_id] = trace.generation(
                name=f"LM-{getattr(instance, 'model_name', instance.__class__.__name__)}",
                input=inputs,
                model=getattr(instance, 'model_name', 'unknown'),
                metadata={"lm_class": instance.__class__.__name__}
            )
            return

        parent_obj = self.calls.get(parent_call_id)
        if parent_obj is None:
            logger.warning("Parent trace not found for LM call. Creating a standalone trace.")
            trace = self.langfuse.trace(name="Orphan-LM-Call", input=inputs, tags=["lm_call"])
            generation = trace.generation(
                name=f"LM-{getattr(instance, 'model_name', instance.__class__.__name__)}",
                input=inputs,
                model=getattr(instance, 'model_name', 'unknown'),
                metadata={"lm_class": instance.__class__.__name__}
            )
            self.calls[call_id] = generation
            return

        # If parent_obj is a trace or any other observation, we can create a generation under it.
        try:
            generation = parent_obj.generation(
                name=f"LM-{getattr(instance, 'model_name', instance.__class__.__name__)}",
                input=inputs,
                model=getattr(instance, 'model_name', 'unknown'),
                metadata={"lm_class": instance.__class__.__name__}
            )
            self.calls[call_id] = generation
        except Exception as e:
            logger.warning(f"Failed to create generation for LM start: {e}")

    def on_lm_end(
            self,
            call_id: str,
            outputs: Optional[Dict[str, Any]],
            exception: Optional[Exception] = None
    ):
        generation = self.calls.get(call_id)
        if generation is None:
            return

        usage = None
        print(outputs)
        # Check if outputs contain a usage field with token counts
        if outputs and "usage" in outputs:
            # Assume the LM returns usage in OpenAI style keys: prompt_tokens, completion_tokens, total_tokens
            print("usage_data", outputs["usage"])
            usage_data = outputs["usage"]
            prompt_tokens = usage_data.get("prompt_tokens")
            completion_tokens = usage_data.get("completion_tokens")
            total_tokens = usage_data.get("total_tokens")

            if prompt_tokens is not None and completion_tokens is not None and total_tokens is not None:
                # Convert them to the OpenAI-style usage keys that Langfuse expects
                usage = {
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens,
                    "totalTokens": total_tokens
                }

        try:
            if exception:
                generation.end(
                    output=outputs if outputs else {},
                    level="ERROR",
                    status_message=str(exception),
                    usage=usage
                )
            else:
                generation.end(
                    output=outputs if outputs else {},
                    usage=usage
                )
        except Exception as e:
            logger.warning(f"Failed to end generation for LM end: {e}")

        # Clean up reference
        self.calls.pop(call_id, None)