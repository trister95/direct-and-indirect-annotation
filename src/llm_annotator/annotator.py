import ujson as json
import os
import re
import asyncio
from func_timeout import func_set_timeout
from openai import RateLimitError
from langchain_core import prompts, output_parsers
from langchain_openai import ChatOpenAI
from aiolimiter import AsyncLimiter

class Annotator:
    def __init__(self, engine: str = 'gpt-3.5-turbo', config_name: str = 'default', dataset: str = None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, 'configs', f'{config_name}.json')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        self.dataset = dataset or config['dataset']
        self.task = config['task']
        self.description = config['description']
        self.guidance = config['guidance']
        self.input_format = config['input_format']
        self.output_format = config['output_format']
        self.struct_format = config['struct_format']

        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=engine)

        # Setup prompt and output parsers

        self.prompt_template = prompts.ChatPromptTemplate.from_messages([
            ("system", self.description.replace("{", "{{").replace("}", "}}")),  # Escaping the braces
            ("system", self.guidance),
            ("user", "{input}")
        ])

        self.output_parser = output_parsers.StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser

        # Setup for enrichment strategy
        self.enrichment_description = config["enrichment_description"] #must put in config
        self.enrichment_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo-0125")
        self.enrichment_prompt_template = prompts.ChatPromptTemplate.from_messages([
            ("system", self.enrichment_description), 
            ("user", "{input}")
        ])
        self.enrichment_output_parser = output_parsers.StrOutputParser()
        self.enrichment_chain = self.enrichment_prompt_template | self.enrichment_llm | self.enrichment_output_parser
        self.limiter = AsyncLimiter(1000)



    def generate_prompt(self, sample, demo=None):
        to_annotate = self.input_format.format(json.dumps(sample['text']))
        if demo:
            demo_annotations = "\n".join(
                f"{self.input_format.format(json.dumps(d['text']))}\n{self.output_format.format(json.dumps(d['labels']))}" for d in demo
            )
            return f"Here are some examples:\n{demo_annotations}\n\nPlease now annotate the following input:\n{to_annotate}"
        else:
            return f"Please annotate the following input:\n{to_annotate}"
    
    @func_set_timeout(60)
    def online_annotate(self, sample, demo=None):
        annotation_prompt = self.generate_prompt(sample, demo)
        retry_count = 0  # Initialize retry counter

        while retry_count < 3:  # Allow up to 3 attempts (initial + 2 retries)
            try:
                response = self.chain.invoke({"input": annotation_prompt})
                return self.postprocess(response)

            except RateLimitError:
                print("Rate limit exceeded. Please wait and try again.")
                print(f"Problem was with: {annotation_prompt}")
                return None

            except Exception as e:
                print(f"Error during annotation: {e}")
                print(f"Problem was with: {annotation_prompt}")
                retry_count += 1  # Increment retry counter

                if retry_count == 3:
                    print("Max retries reached. Aborting operation.")
                    return None

                print("Retrying...")

        return None
    
    def postprocess(self, result):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.dirname(os.path.dirname(dir_path))
        meta_path = os.path.join(dir_path, f'data/{self.dataset}/meta.json')
        with open(meta_path, 'r') as file:
            meta = json.load(file)
        tagset = meta['tagset']
        list_pattern = r"\[([^\]]*)\]"
        match = re.search(list_pattern, result)
        if match:
            # Convert the matched string into a list if it is not empty, otherwise create an empty list
            extracted_result = eval(f"[{match.group(1)}]") if match.group(1) else []
        else:
            print("No list found.")

        outputs = []
        for entity in extracted_result:
            if not isinstance(entity, dict):
                continue
            if 'type' not in entity or 'span' not in entity:
                continue
            if entity['type'] in tagset:
                outputs.append(entity)
        return outputs

    @staticmethod
    async def string_to_bool(s):
        if s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        else:
            raise ValueError("Input must be 'true' or 'false'")        

    async def process_sentence_async(self, sample):
        async with self.limiter:
            annotation_prompt = self.input_format.format(json.dumps(sample['text']))
            retry_count = 0  # Initialize retry counter

            while retry_count < 3:  # Allow up to 3 attempts (initial + 2 retries)
                try:
                    response = await asyncio.wait_for(self.enrichment_chain.ainvoke({"input": annotation_prompt}), timeout=60)
                    match = re.search(r"\b(True|False)\b", response)
                    if match:
                        return await self.string_to_bool(match.group(0))
                    else:
                        return False

                except RateLimitError:
                    print("Rate limit exceeded. Please wait and try again.")
                    print(f"Problem was with: {annotation_prompt}")
                    return None

                except Exception as e:
                    print(f"Error during annotation: {e}")
                    print(f"Problem was with: {annotation_prompt}")
                    retry_count += 1  # Increment retry counter

                    if retry_count == 3:
                        print("Max retries reached. Aborting operation.")
                        return None

                    print("Retrying...")
            return None

    async def process_batch_llm_async(self, batch):
        return await asyncio.gather(*[self.process_sentence_async(sentence) for sentence in batch])
    
if __name__ == '__main__':
    annotator = Annotator(engine='gpt-3.5-turbo', config_name='en_conll03')
    sample = {"tokens":["因","盛","产","绿","竹","笋","，","被","誉","为","「","绿","竹","笋","的","故","乡","」","的","八","里","，","就","像","台","湾","许","多","大","大","小","小","遥","远","的","乡","镇","，","在","期","待","与","失","落","中","，","承","载","着","生","活","必","需","的","悲","苦","与","欢","乐","，","并","由","于","位","处","边","陲","，","担","负","着","众","人","不","愿","承","受","之","重","。"],"tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-GPE","E-GPE","O","O","O","B-GPE","E-GPE","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O"],"text":"因盛产绿竹笋，被誉为「绿竹笋的故乡」的八里，就像台湾许多大大小小遥远的乡镇，在期待与失落中，承载着生活必需的悲苦与欢乐，并由于位处边陲，担负着众人不愿承受之重。","labels":[{"span":"台湾","type":"GPE"},{"span":"八里","type":"GPE"}],"id":"23549"}
    demo = [sample]
    print(annotator.online_annotate(sample))

