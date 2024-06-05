from typing import Dict
from pydantic import BaseModel
import torch
import os
from transformers import BertTokenizer,AlbertTokenizer,AutoTokenizer, AutoModelForQuestionAnswering ,BertForQuestionAnswering, AlbertForQuestionAnswering
import torch.nn as nn
import torch.optim as optim
import json

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class NLPManager:
    def __init__(self):
        # initialize the model here
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #model_name='bert-large-uncased-whole-word-masking-finetuned-squad'
        #model = BertForQuestionAnswering.from_pretrained(model_name) 
        #tokenizer = BertTokenizer.from_pretrained(model_name)
        
     
        # Pointing self.infer to the model itself
        #self.infer = self.model
        
        #model_directory = os.getenv("MODEL_PATH", "/workspace/models_export")        
        #self.model.to(self.device)
        #self.tokenizer = BertTokenizer.from_pretrained(model_name)        
        #self.model = BertForQuestionAnswering.from_pretrained(model_directory, device_map=self.devic)

        
        # Placeholder for processor
        #self.processor =  BertForQuestionAnswering.from_pretrained(model_directory, device_map=self.devic)  
        
        '''
        # self.infer = 
        self.infer = 
        self.infer.to(self.device)

    
        self.model =
        self.processor = 
        '''
        
        pass

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        model_name='bert-large-uncased-whole-word-masking-finetuned-squad'
        #model = BertForQuestionAnswering.from_pretrained(model_name) 
        #tokenizer = BertTokenizer.from_pretrained(model_name)
        

        #directory = "../model_export"

        # Create the directory if it doesn't exist
        #os.makedirs(directory, exist_ok=True)

        # Specify the file path
        #PATH = os.path.join(directory, "model.safetensors")

        
        #net = Net()
        #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        # Specify a path
        
        #current_directory = os.getcwd()
        #print("Current directory:", current_directory)
        
        
        #PATH = "../model_export/model.safetensors"
        #PATH = "../model_export2/"
        currentScriptPath=os.path.dirname(__file__)
        #print("Model File path exist:" , os.path.isfile(PATH))
        print("Current File Directory",currentScriptPath)
        modelPath=os.path.join(currentScriptPath,"../model_export2/")
        tokenzierPath=os.path.join(currentScriptPath,"../model_export2/")
        # Save
        #torch.save(net, PATH)

        # Load
        #model = torch.load(PATH)
        #model.eval()
        
        #model_directory = os.getenv("MODEL_PATH", "/workspace/models_export")        
        #self.model.to(self.device)
        
        
        tokenizer = BertTokenizer.from_pretrained(tokenzierPath)        
        #model = BertForQuestionAnswering.from_pretrained(PATH, device_map=self.device)     
        model=BertForQuestionAnswering.from_pretrained(modelPath,use_safetensors=True)      
        print(str(tokenizer))
        print(str(model))
        
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
    nlpmanager=NLPManager()
    with open("/home/jupyter/til-team-33/NLP-Test/small_nlp.jsonl", "r") as f:     
        instances = [json.loads(line.strip()) for line in f if line.strip() != ""]
    for currentinstance in instances:        
        print(nlpmanager.qa(currentinstance['transcript']))


def qa_bertmodel(question,answer_text,model,tokenizer):
  inputs = tokenizer.encode_plus(question, answer_text, add_special_tokens=True, return_tensors="pt")
  input_ids = inputs["input_ids"].tolist()[0]

  text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
  print(text_tokens)
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

  print('Answer: "' + answer + '"')
  return answer

def number_to_words(number):
    number_mapping = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    return ' '.join([number_mapping[char] for char in number])

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
