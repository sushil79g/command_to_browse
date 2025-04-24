import sys
from textwrap import dedent
from typing import Union, List
from agno.agent import Agent
from agno.models.ollama import Ollama
from pydantic import BaseModel, Field
from loguru import logger




class ExplainerOutput(BaseModel):
    query: str = Field(..., description="Query passed to LLM")
    steps: Union[List[str], None] = Field(..., description="List of steps to reach the goal")


class VerifierOutput(BaseModel):
    query: str = Field(..., description="Initial user prompt")
    steps: str = Field(..., description="Steps input for verification")
    evaluation: str = Field(..., description="Evaluation details from verifier")
    verdict: str = Field(..., description="Verdict: Valid Plan or Invalid Plan")
    explainer: str = Field(..., description="Explanation for the verdict")
    final_step: str = Field(..., description="Final improved steps with added context")


class WebTaskPlanner:
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
        self.agent = self._create_explainer_agent()
        self.verifier_agent = self._create_verifier_agent()

    def _create_explainer_agent(self) -> Agent:
        return Agent(
            model=Ollama(id="qwen2.5:32b-instruct-q4_K_M"),
            instructions=dedent("""
                You are a Web Task Planner Agent. Your job is to take a user's prompt about a web-based task and break it down into a clear, step-by-step plan to achieve the goal.

                Guidelines:
                - Understand the user's objective.
                - Identify the necessary steps to accomplish the task.
                - Present the steps in a numbered list.
                - Be concise and clear in your instructions.
                - If applicable, include any limits or constraints mentioned.
            """),
            response_model=ExplainerOutput,
            markdown=True,
        )

    def _create_verifier_agent(self) -> Agent:
        return Agent(
            model=Ollama(id="gemma:7b-instruct"),
            instructions=dedent("""
                You are a Response Verifier Agent. Your task is to evaluate whether a given step-by-step plan correctly accomplishes the specified web-based task.

                Guidelines:
                - Analyze the user's original prompt to understand the intended goal.
                - Review the provided steps to determine if they logically and effectively achieve the goal.
                - Identify any missing, incorrect, or unnecessary steps.
                - Provide a verdict: "Valid Plan" or "Invalid Plan".
                - If invalid, explain the issues and suggest corrections.
            """),
            response_model=VerifierOutput,
            markdown=True,
        )

    def plan_task(self, user_prompt: str) -> Union[str, None]:
        retries = 0
        query = user_prompt
        final_output = None

        while retries < self.max_retries:
            logger.info(f"Attempt {retries + 1} of {self.max_retries}")
            final_output = self._generate_and_evaluate(query)

            if final_output.content.verdict.lower() == "valid plan":
                logger.success("Received a valid plan.")
                break
            else:
                logger.warning("Invalid plan. Retrying with feedback...")
                query = self._construct_retry_prompt(user_prompt, final_output)
                retries += 1

        if final_output:
            logger.info("Returning final steps.")
            return final_output.content.steps

        logger.error("Max retries exceeded. No valid plan generated.")
        return None

    def _generate_and_evaluate(self, prompt: str) -> VerifierOutput:
        logger.debug(f"Sending to Explainer: {prompt}")
        explainer_response = self.agent.run(prompt)
        steps = explainer_response.content.steps
        logger.debug(f"Explainer response: {steps}")

        evaluation_input = f'User Prompt: "{prompt}"\n\nProposed Steps:\n{steps}'
        logger.debug(f"Sending to Verifier: {evaluation_input}")
        verifier_response = self.verifier_agent.run(evaluation_input)

        logger.debug(f"Verifier verdict: {verifier_response.content.verdict}")
        logger.debug(f"Evaluation: {verifier_response.content.evaluation}")
        logger.debug(f"Explainer notes: {verifier_response.content.explainer}")

        return verifier_response

    def _construct_retry_prompt(self, base_prompt: str, verifier_output: VerifierOutput) -> str:
        return (
            f"{base_prompt} Steps predicted are: {verifier_output.content.steps}\n"
            f"Evaluation: {verifier_output.content.explainer}\n"
            "Please revise the steps considering this feedback."
        )


# Example run
if __name__ == "__main__":
    user_prompt = "Go to https://coinmarketcap.com/ and list all the coins that are priced between $30,000 and $1,000,000 today."
    planner = WebTaskPlanner(max_retries=5)
    result_steps = planner.plan_task(user_prompt)

    if result_steps:
        logger.info(f"Final Steps:\n{result_steps}")
    else:
        logger.warning("Task planning did not succeed.")
