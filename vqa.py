from dotenv import load_dotenv
import os
from openai import OpenAI
from prompt import USER_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_TAKE_CONTEXT_QUESTION, NEW_CONTEXT_QUESTION

load_dotenv()

class VQA:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        
    def refine(self, 
               discription: str,
               question: str,
               system_prompt: str = NEW_CONTEXT_QUESTION):
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": USER_PROMPT.format(discription, question)}
                ],
                temperature=0,
            )
            return response.choices[0].message.content.split("->")[-1].strip()

        except:
            return False
        
v = VQA()
a = v.refine(discription="Authorities stood around a person wearing a light yellow-brown shirt with a tattoo",
             question="Ask in the scene where there is 1 person holding a phone to film, how many people are in the frame?")
print(a)