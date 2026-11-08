import transformers
import torch

from LLMs.LLM import LLM
from utils.logger import setup_logger
from constants.constants import HUGGING_FACE_LLM_LOG_FILE, HUGGING_FACE_LLM_LOG_NAME

# Models already loaded, by model name, so agents that use the same model share one copy in GPU memory
_LOADED_MODELS = {}


class HuggingFaceLLM(LLM):
    """
        HuggingFaceLLM class for generating text using a specified model with configurable temperature from Hugging Face.
        Inherits from the base LLM class.
    """

    def __init__(self, model_name: str, temperature: float, token: str, **kwargs):
        """
        Initialize the HuggingFaceLLM model with the specified model name and temperature.

        :param model_name: Name of the model to use for text generation.
        :param temperature: Temperature parameter for controlling randomness in generation.
        :param token: Token to use for the model.
        """
        super().__init__(model_name, temperature, token)

        self.__generate_flags = kwargs

        self.__logger = setup_logger(HUGGING_FACE_LLM_LOG_NAME, HUGGING_FACE_LLM_LOG_FILE)
        self.__logger.info(f"Hugging Face LLM model initialized successfully with model: {model_name}")

        try:
            # Load the model once per model name and reuse it for every agent that uses it
            if self.model_name not in _LOADED_MODELS:
                model = transformers.pipeline("text-generation", model=self.model_name,
                                              model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",
                                              token=self.token)
                tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
                _LOADED_MODELS[self.model_name] = (model, tokenizer)
            self.__model, self.__tokenizer = _LOADED_MODELS[self.model_name]
        except Exception as e:
            self.__logger.error(f"Error initializing hugging face model: {e}")
            raise

    def generate_prompt(self, user: str, system: str, max_tokens: int, clean: bool = True,
                        force_max_tokens: bool = False) -> str:
        """
        Generate a text document based on user and system prompts.

        :param user: The user prompt.
        :param system: The system prompt.
        :param max_tokens: Maximum number of tokens for the generated document.
        :param clean: Whether to clean the document of extraneous text or not.
        :param force_max_tokens: Whether to manually restrict the number of tokens in the generated document.

        :return: The generated document as a string.
        """
        try:
            # Construct the message structure
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]

            # Generate text based on device type
            try:
                result = self.__model(messages, max_new_tokens=max_tokens, temperature=self.temperature, do_sample=True,
                                      **self.__generate_flags)[0]['generated_text'][-1]["content"]
            except Exception as e:
                # Modify the messages for the second attempt
                messages = [{"role": "user", "content": f"{system} {user}"}]

                try:
                    result = self.__model(messages, max_new_tokens=max_tokens, temperature=self.temperature,
                                          do_sample=True, **self.__generate_flags)[0]['generated_text'][-1]["content"]
                except Exception as e:
                    self.__logger.error(f"Error in generating prompt on second attempt: {e}")
                    raise

            # Clean the generated document
            if clean:
                cleaned_result = self.clean_document(result, max_tokens, self)

                # Trim the generated document to max_tokens length
                if force_max_tokens:
                    cleaned_result = self.__trim_tokens(cleaned_result, max_tokens)

                return cleaned_result, result
            else:
                return result
        except Exception as e:
            self.__logger.error(f"Error in generating prompt: {e}")
            raise

    def __trim_tokens(self, input_string: str, max_tokens: int) -> str:
        """
        Trims the input string to ensure that it contains no more than max_tokens tokens.

        :param input_string: The string to be tokenized and trimmed.
        :param max_tokens: The maximum number of tokens allowed.
        :return: The trimmed string with no more than max_tokens tokens.
        """
        # Tokenize the input string
        tokens = self.__tokenizer.encode(input_string, return_tensors="pt")[0]

        # Check if the number of tokens exceeds max_tokens
        if len(tokens) > max_tokens:
            # Trim the tokens to max_tokens length
            tokens = tokens[:max_tokens]

        # Decode the tokens back into a string
        trimmed_string = self.__tokenizer.decode(tokens, skip_special_tokens=True)

        return trimmed_string
