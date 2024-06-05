from typing import Dict
from pydantic import BaseModel
import torch
import os
from transformers import BertTokenizer,AlbertTokenizer,AutoTokenizer, AutoModelForQuestionAnswering ,BertForQuestionAnswering, AlbertForQuestionAnswering
import torch.nn as nn
import torch.optim as optim
import json



class NLPManager:
    def __init__(self):
        # initialize the model here
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pass

    def qa(self, context: str) -> Dict[str, str]:

        model_name='bert-large-uncased-whole-word-masking-finetuned-squad'
 
        currentScriptPath=os.path.dirname(__file__)

        print("Current File Directory",currentScriptPath)
        modelPath=os.path.join(currentScriptPath,"../model_export2")
        tokenzierPath=os.path.join(currentScriptPath,"/model_export2/")
 
        try:            
            print("Current File Directory",currentScriptPath)
            print("Path1:",tokenzierPath)
            tokenizer = BertTokenizer.from_pretrained(tokenzierPath)  
        except:
            print("Path 1 error")
            try:
                tokenzierPath=os.path.join(currentScriptPath,"/model_export2")
                print("Current File Directory",currentScriptPath)
                print("Path2:",tokenzierPath)
                tokenizer = BertTokenizer.from_pretrained(tokenzierPath)  
            except:
                print("Path 2 Error")
                try:
                    tokenzierPath=os.path.join(currentScriptPath,"/model_export2/model.safetensors")
                    print("Current File Directory",currentScriptPath)
                    print("Path3:",tokenzierPath)
                    tokenizer = BertTokenizer.from_pretrained(tokenzierPath) 
                except:
                    print("Path 3 Error")
                    try:
                        tokenzierPath=os.path.join(currentScriptPath,"../model_export2")
                        print("Current File Directory",currentScriptPath)
                        print("Path4:",tokenzierPath)
                        tokenizer = BertTokenizer.from_pretrained(tokenzierPath) 
                    except:
                        print("Path 4 Error")
                        try:
                            tokenzierPath=os.path.join(currentScriptPath,"../model_export2/model.safetensors")
                            print("Current File Directory",currentScriptPath)
                            print("Path5:",tokenzierPath)
                            tokenizer = BertTokenizer.from_pretrained(tokenzierPath) 
                        except:
                            print("Path 5 Error")
                            try:
                                tokenzierPath=os.path.join(currentScriptPath,"model_export2")
                                print("Current File Directory",currentScriptPath)
                                print("Path 6:",tokenzierPath)
                                tokenizer = BertTokenizer.from_pretrained(tokenzierPath)                          
                            except:
                                print("Path 6 Error")
                                try:
                                    tokenzierPath=os.path.join(currentScriptPath,"model_export2/model.safetensors")
                                    print("Current File Directory",currentScriptPath)
                                    print("Path 7:",tokenzierPath)
                                    tokenizer = BertTokenizer.from_pretrained(tokenzierPath)    
                                except:
                                     print("Path 7 error")
                                else:
                                     print("Path 7 ok")
                            else:
                                print("Path 6 ok")
                        else:
                            print("Path 5 ok")
                        
                    else:
                        print("Path 4 ok")
                    
                else:
                    print("Path 3 Ok")
            else:   
                print("Path 2 Ok")
        
        else:
            print("Path 1 Ok")
            
        print("Final Path:",tokenzierPath)
            
            
        tokenizer = BertTokenizer.from_pretrained(tokenzierPath)        

        model=BertForQuestionAnswering.from_pretrained(tokenzierPath,use_safetensors=True)      
        
        toolQuestion="Description of the weapon used." #What is the weapon used?
        headingQuestion="What is the heading mentioned?"
        targetQuestion="Description of the enemy plane" #Description of the adversary flying object.
        transcriptStr=context
        
        toolAnswer=qa_bertmodel(toolQuestion,transcriptStr,model,tokenizer)
        headingAnswer=words_to_number(qa_bertmodel(headingQuestion,transcriptStr,model,tokenizer))
        targetAnswer=qa_bertmodel(targetQuestion,transcriptStr,model,tokenizer)
        
        toolAnswer=remove_spacing_with_hyphens(toolAnswer)
        headingAnswer=remove_spacing_with_hyphens(headingAnswer)
        targetAnswer=remove_spacing_with_hyphens(targetAnswer)
        
        return {"heading": headingAnswer, "tool": toolAnswer, "target": targetAnswer}
    
def main():
    print("Running Main")  
    nlpmanager=NLPManager()
    with open("/home/jupyter/til-team-33/NLP-Test/small_nlp.jsonl", "r") as f:     
        instances = [json.loads(line.strip()) for line in f if line.strip() != ""]
    for currentinstance in instances:      
        print("Transcript: ", currentinstance['transcript'] )
        print("Final Answer for Key ", currentinstance['key'], " ; " ,nlpmanager.qa(currentinstance['transcript']), "\n\n")
    

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
    print("Transcript: ", answer_text)
    print("Question: ",question," ; Answer: ", answer)
    return answer

def number_to_words(number):
    number_mapping = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'niner'
    }
    return ' '.join([number_mapping[char] for char in number])

def words_to_number(wordStr):
    number_mapping = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'niner': '9', 'nine': '9'
    }
    words = wordStr.split()  # Split the input string into words
    numbers = [number_mapping[word.lower()] for word in words if word.lower() in number_mapping]
    return ''.join(numbers)

def remove_spacing_with_hyphens(text):
    # Split the text on hyphens
    parts = text.split('-')
    
    # Remove spaces from each part and join them back together with hyphens
    corrected_text = '-'.join(part.strip() for part in parts)
    
    return corrected_text

if __name__ == "__main__":
    main()