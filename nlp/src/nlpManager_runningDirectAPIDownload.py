from typing import Dict
from pydantic import BaseModel
import torch
import os
import json
from transformers import BertTokenizer,AlbertTokenizer,AutoTokenizer, AutoModelForQuestionAnswering ,BertForQuestionAnswering, AlbertForQuestionAnswering

print("Top")

class NLPManager:
    def __init__(self):
        # initialize the model here
        print("initing1")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pass
    

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        model_name='bert-large-uncased-whole-word-masking-finetuned-squad'
        model = BertForQuestionAnswering.from_pretrained(model_name) 
        tokenizer = BertTokenizer.from_pretrained(model_name)

        toolQuestion="What is the weapon used?"
        headingQuestion="What is the heading mentioned?"
        targetQuestion="What is the target?"
        transcriptStr=context

        toolAnswer=qa_bertmodel(toolQuestion,transcriptStr,model,tokenizer)
        headingAnswer=words_to_number(qa_bertmodel(headingQuestion,transcriptStr,model,tokenizer))
        targetAnswer=qa_bertmodel(targetQuestion,transcriptStr,model,tokenizer)

        return {"heading": headingAnswer, "tool": toolAnswer, "target": targetAnswer}
    
    
def main():
    print("Running Main")  
    data_2d_array = []
    results = []
    nlp_manager = NLPManager()
    with open("/home/jupyter/til-team-33/NLP-Test/nlp_new/small_nlp.jsonl", "r") as f:     
        instances = [json.loads(line.strip()) for line in f if line.strip() != ""]
    for instance in instances:
        
        _results=(nlp_manager.qa(context=instance['transcript']))
        print("\n\nPrinting Instance" + str(instance))
        print("Printing Results" + str(_results) + "\n\n")
        results.extend(
            [
                {
                    "key": instance["key"],
                    "truth": {
                        field: instance[field]
                        for field in ("heading", "target", "tool")
                    },
                    "prediction": _results,
                }
                for i in range(len(instance))
            ]
        )
    
        #print(results)
    #data_2d_array.append([instance['key'], instance['transcript'], instance['tool'], instance['heading'], instance['target']])

    
    
    
    

def qa_bertmodel(question,answer_text,model,tokenizer):
  inputs = tokenizer.encode_plus(question, answer_text, add_special_tokens=True, return_tensors="pt")
  input_ids = inputs["input_ids"].tolist()[0]

  text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
  #print(text_tokens)
  outputs = model(**inputs)
  answer_start_scores=outputs.start_logits
  answer_end_scores=outputs.end_logits

  answer_start = torch.argmax(
      answer_start_scores
  )  # Get the most likely beginning of answer with the argmax of the score
  answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

  # Combine the tokens in the answer and print it out.""
  answer = answer.replace("#","")

  #print(question+' : "' + answer + '"')
  return answer



def words_to_number(wordStr):
    number_mapping = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
    }
    words = wordStr.split()  # Split the input string into words
    numbers = [number_mapping[word.lower()] for word in words if word.lower() in number_mapping]
    return ''.join(numbers)

if __name__ == "__main__":
    main()