import pandas as pd

from utils.logger import setup_logger
from players.player import Player
from LLMs.LLM import LLM
from competition.prompt_manager import PromptManager
from constants.constants import LLM_PLAYER_LOG_FILE, LLM_PLAYER_LOG_NAME


class LLMPlayer(Player):
    """
        A player in the game that uses Large Language Models (LLMs)
        to generate documents and process feedback across rounds.
    """

    def __init__(self, name: str, character: str, llm: LLM, prompt_format: str, query: str, query_id: int,
                 init_document: str, feedback_func: callable):
        """
        Initialize a Player instance.

        :param name: Name of the player.
        :param character: Character/persona the player will adopt.
        :param prompt_format: Format string for prompts.
        :param query: The query the player will be working with.
        :param query_id: The ID of the query.
        :param init_document: Initial document text for the player.
        :param feedback_func: Function to generate feedback for the player.
        """
        super().__init__(name, character, prompt_format, query, query_id, init_document, feedback_func)
        self.__llm = llm
        self.prompt_manager = PromptManager(prompt_format)
        self.__logger = setup_logger(LLM_PLAYER_LOG_NAME, LLM_PLAYER_LOG_FILE)

    def generate_document(self, max_tokens: int, init_doc: str = None,
                          force_max_tokens: bool = False, ) -> tuple:
        """
        Generate a document based on the query and character using the specified LLM.

        :param max_tokens: Maximum number of tokens for the generated document.
        :param init_doc: Initial document text, used in the first round.
        :param force_max_tokens: Whether to manually restrict the number of tokens in the generated document.

        :return: The generated document.
        """
        try:
            non_cleaned_document, user_prompt, system_prompt = None, None, None

            if self.round > 1:
                user_prompt = self.prompt_manager.build_user_prompt(self.__pairwise_feedback, self.__all_feedback,
                                                                    self.query)
                system_prompt = self.prompt_manager.build_system_prompt(self.query, self.document, self.character)
                self.document, non_cleaned_document = self.__llm.generate_prompt(user_prompt, system_prompt, max_tokens,
                                                                          force_max_tokens=force_max_tokens)
            else:
                # Game.generate_documents does not pass init_doc, so fall back to the player's initial document
                if init_doc is None:
                    init_doc = self.init_document
                system_prompt = self.prompt_manager.build_system_prompt(self.query, init_doc, self.character)
                self.document, non_cleaned_document = self.__llm.generate_prompt("", system_prompt, max_tokens,
                                                                          force_max_tokens=force_max_tokens)

            return self.document, non_cleaned_document, user_prompt, system_prompt
        except Exception as e:
            self.__logger.error(f"Error generating document: {e}")
            raise

    def generate_feedback(self, feedback: pd.DataFrame) -> None:
        """
        Generate feedback for the next round based on the provided feedback.

        :param feedback: Feedback for the current round.
        """
        self.round += 1
        self.__pairwise_feedback, self.__all_feedback = self.feedback_func(feedback, self.name, self.round)
