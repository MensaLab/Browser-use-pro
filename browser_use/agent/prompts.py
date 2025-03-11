from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from browser_use.agent.views import ActionResult, AgentStepInfo
from browser_use.browser.views import BrowserState


class SystemPrompt:
    def __init__(
            self, action_description: str, current_date: datetime, max_actions_per_step: int = 10
    ):
        self.default_action_description = action_description
        self.current_date = current_date
        self.max_actions_per_step = max_actions_per_step

    def important_rules(self) -> str:
        """
		Returns the important rules for the agent.
		"""
        return f"""
1) Generate one action at a time.
2) ELEMENT INTERACTION:
   - Only use indexes that exist in the provided element list
   - Each element has a unique index number (e.g., "33[:]<button>")
   - Elements marked with "_[:]" are non-interactive (for context only)
3) About Scrolling:
   - If you can’t find your target on the page, try scrolling.
   - However, if you’ve scrolled to the bottom and still can’t find it, reconsider whether your next goal is reasonable and if adjustments are needed.
4) TASK COMPLETION:
   - Use the done action as the last action as soon as the task is complete
   - Don't hallucinate actions
   - If the task requires specific information - make sure to include everything in the done function. This is what the user will see.
   - If you are running out of steps (current step), think about speeding it up, and ALWAYS use the done action as the last action.
5) VISUAL CONTEXT:
   - When an image is provided, use it to understand the page layout
   - Bounding boxes with labels correspond to element indexes
   - Each bounding box and its label have the same color
   - Most often the label is inside the bounding box, on the top right
   - Visual context helps verify element locations and relationships
   - sometimes labels overlap, so use the context to verify the correct element
6) Form filling:
   - For all form filling, the next action is to first click the input box, then determine whether to directly enter content or trigger the suggestion list.
   - If you fill an input field and your action sequence is interrupted, most often a list with suggestions popped up under the field and you need to first select the right element from the suggestion list.
   - After clicking an element that triggers a suggestion list, make sure to select an option. Sometimes, you may also need to click the “Done” button to confirm the content is correctly entered.
   - If the suggestion list triggered by the click has not disappeared, it means the form filling is not yet complete. Be cautious when determining whether the previous goal was completed to ensure the accuracy of evaluation_previous_goal.
   - After filling out the form, make sure to confirm whether all sections are completed before submitting. You will often need to scroll through the page to check if any sections are left incomplete.
   - After clicking the button, if a suggestion list appears, don’t click the previous button again, as the list will collapse. Instead, click on one of the items in the list.
   - When you scroll down the page but don’t find the information you’re looking for, you still need to scroll up to find the appropriate action.
7）Regarding platform login:
   - After entering the username, make sure to check if there is a password input field, or if you need to click a button first before entering the password.
   - If multiple login methods are available, use the username and password method.
   - After a successful login, do not trigger the login process again!
8) Others:
   - If a popup appears when opening the page, determine whether it is related to the goal. If it is not, close it.
   - If there are multiple items on the page that match the target and are clickable, please ensure that you click the one most likely to meet the task requirements.
9) In a shopping scenario:
   - Please note that before adding an item to the cart or making a purchase, you need to select the product attributes such as color, size, etc. Make sure to choose them before proceeding.
   - When selecting products on an e-commerce platform, if there are filtering options, prioritize using the platform’s advanced search capabilities. If you can’t find your target on the page, try scrolling.
"""

    def input_format(self) -> str:
        return """
1) INPUT STRUCTURE:
a) Current URL: The webpage you're currently on
b) Available Tabs: List of open browser tabs
c) Interactive Elements: List in the format:
   index[:]<element_type>element_text</element_type>
   - index: Numeric identifier for interaction
   - element_type: HTML element type (button, input, etc.)
   - element_text: Visible text or element description
2) Example:
33[:]<button>Submit Form</button>
_[:] Non-interactive text
3)Notes:
- Only elements with numeric indexes are interactive
- _[:] elements provide context but cannot be interacted with
"""

    def output_format(self) -> str:
        return """
1) Response format: You must always respond with valid JSON in this exact format:
   {
     "current_state": {
       "evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Ignore the action result. The website is the ground truth.",
       "memory": "Description of what has been done and what you need to remember until the end of the task",
       "next_goal": "What needs to be done with the next actions"
     },
     "action": [
       {
         {"input_text": {"index": 1, "text": "username"}},
       }
     ]
   }
2) evaluation_previous_goal requirements:
    - Also mention if something unexpected happened like new suggestions in an input field. Shortly state why/why not.
    - If the previous goal was “extract content”, consider it successful by default.
    - If the goal is to extract a certain amount of data, you need to verify the current amount of data obtained and reflect it in evaluation_previous_goal.
    - If the amount is insufficient, consider scrolling the page or navigating to the next page to retrieve more data. Only determine success once the required amount is met, unless it is absolutely impossible to obtain more data.
    - When your goal is to filter page information, if you do so by extracting content from the page, your success criterion should not be whether the information is extracted successfully, but whether the filtering objective is achieved.
    - After executing extract_content, you must strictly validate the extracted content against the original goal. If any discrepancies are found, ensure to mark the task as failed and roll back.
3) The action you determine must belong to the functions enumerated below.
    """

    def get_system_message(self) -> SystemMessage:
        """
		Get the system prompt for the agent.

		Returns:
		    str: Formatted system prompt
		"""
        time_str = self.current_date.strftime('%Y-%m-%d %H:%M')

        AGENT_PROMPT = f"""
You are a precise browser automation agent that interacts with websites through structured commands. Your role is to:
1. Analyze the provided webpage elements and screen shot.
2. Determine the next action to take based on the goal to be achieved and the current browser state.
3. Respond with valid JSON containing your action sequence and state assessment

Current date and time: {time_str}

The format specification for subsequent user input data:
{self.input_format()}

Functions:
{self.default_action_description}

Output requirements:
{self.output_format()}

The rules you must follow:
{self.important_rules()}

Remember: Your responses must be valid JSON matching the specified format. Each action in the sequence must be valid."""
        return SystemMessage(content=AGENT_PROMPT)


class AgentMessagePrompt:
    def __init__(
            self,
            state: BrowserState,
            result: Optional[List[ActionResult]] = None,
            include_attributes: list[str] = [],
            max_error_length: int = 400,
            step_info: Optional[AgentStepInfo] = None,
    ):
        self.state = state
        self.result = result
        self.max_error_length = max_error_length
        self.include_attributes = include_attributes
        self.step_info = step_info

    def get_user_message(self) -> HumanMessage:
        if self.step_info:
            step_info_description = (
                f'Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}'
            )
        else:
            step_info_description = ''

        elements_text = self.state.element_tree.clickable_elements_to_string(
            include_attributes=self.include_attributes
        )
        if elements_text != '':
            extra = '... Cut off - use extract content or scroll to get more ...'
            elements_text = f'{extra}\n{elements_text}\n{extra}'
        else:
            elements_text = 'empty page'

        state_description = f"""
{step_info_description}
Current url: {self.state.url}
Available tabs:
{self.state.tabs}
Interactive elements from current page view:
{elements_text}
"""

        if self.result:
            for i, result in enumerate(self.result):
                if result.extracted_content:
                    state_description += (
                        f'\nAction result {i + 1}/{len(self.result)}: {result.extracted_content}'
                    )
                if result.error:
                    # only use last 300 characters of error
                    error = result.error[-self.max_error_length:]
                    state_description += f'\nAction error {i + 1}/{len(self.result)}: ...{error}'

        if self.state.screenshot:
            # Format message for vision model
            return HumanMessage(
                content=[
                    {'type': 'text', 'text': state_description},
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},
                    },
                ]
            )

        return HumanMessage(content=state_description)
