import os,sys,json,re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from Graphix.preprocess_my.common_utils import Preprocessor
from Graphix.preprocess_my.inject_syntax import DEP
from Graphix.utils.graph_utils import GraphProcessor
from Graphix.utils.args import init_args
from Graphix.utils.example import Example
from Graphix.utils.batch import Batch
from Graphix.model.model_utils import Registrable
from Graphix.model.encoder.graph_encoder import *
from typing import Annotated, List
from pydantic import BaseModel, Field

class ClassifierInput(BaseModel):
    question: Annotated[str, Field(description="The question to analyze.")]
    db_id: Annotated[str, Field(description="The database id to answer the question.")]

def classifier(input: ClassifierInput) -> List[str]:
    """
    This function analyzes a question and determines the required SQL components.
    
    Args:
        input (ClassifierInput): Input parameters containing question and database id
    
    Returns:
        List[str]: List of required SQL components
    """
    question = input.question
    db_id = input.db_id
    Example.configuration(method='rgatsql',choice='dev')
    args = init_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.relation_num = len(Example.relation_vocab)
    
    components_list = [
        'select', 'multi-select', 'from', 'join', 'multi-join', 'on', 
        'where', 'where-and', 'where-or', 'where-between', 'where-like', 'where-in',
        'cast', 'case', 'union', 'group by', 'order by', 'having', 'distinct',
        'desc', 'asc', 'limit', 'count', 'sum', 'avg', 'min', 'max'
    ]
    
    model = Registrable.by_name('encoder_text2sql')(args).to(args.device)
    model_path = os.path.join(args.output_path, 'best_model3_learningrate.bin')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    ex = []
    all_outputs = []
    preprocessor = Preprocessor(db_dir="Data\dev_databases\\")
    dep = DEP()
    graphprocessor = GraphProcessor()
    database_table_path = r"Data\dev_databases\dev_tables.json"
    database_schema_path = r"Data\dev_databases\dev_schema.json"
    custom_entities_path = r"Data\stopwords\stopwords_codeversion.json"
    
    with open(custom_entities_path, "r") as f:
        custom_entities = json.load(f)
    
    with open(database_table_path, "r") as f:
        database_table = json.load(f)
        
    with open(database_schema_path, "r") as f:
        database_schema = json.load(f)
        
    for info in database_table:
        if db_id == info['db_id']:
            proper_noun = None
            for custom_entity in custom_entities:
                if custom_entity['db_id'] == db_id:
                    proper_noun = custom_entity['stopwords']
                    
            db = preprocessor.preprocess_database(db=info, custom_entities=proper_noun)
            question_text = question
            question_toks_unclean = re.split(r'[^\w\s_]', question_text)
            question_toks = [item.strip() for item in question_toks_unclean if item.strip()]
            
            question_entry = {
                'db_id': db_id,
                'question': question_text,
                'question_toks': question_toks
            }
            
            entry_after_question_preprocessing = preprocessor.preprocess_question(
                entry=question_entry, 
                db=db, 
                custom_entities=proper_noun
            )
            
            entry_after_schema_link = preprocessor.schema_linking(
                entry=entry_after_question_preprocessing, 
                db=db, 
                custom_entities=proper_noun
            )
            
            entry_after_injection = dep.inject_syntax(entry=entry_after_schema_link)

            for schema in database_schema:
                if entry_after_injection['db_id'] == schema['db_id']:
                    entry = graphprocessor.process_graph_utils(
                        ex=entry_after_injection,
                        db=schema,
                        train_dataset=[],
                        is_tool=True
                    )
                    ex.append(entry)
                    break

            dev_dataset = Example.load_dataset_for_tool(ex, is_tool=True)

    with torch.no_grad():
        for i in range(len(dev_dataset)):
            single_example = [dev_dataset[i]]
            current_batch = Batch.from_example_list(single_example, args.device, train=False, is_tool=True)
            
            outputs = model(current_batch)
            predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy()[0].tolist()
            
            filtered_components = []
            for idx, is_selected in enumerate(predictions):
                component = components_list[idx]
                if is_selected and component not in [
                    'select', 'from'
                ]:
                    filtered_components.append(component)
            
            all_outputs.extend(filtered_components)
    
    if not all_outputs:
        all_outputs = ['select', 'from'] 
        
    return all_outputs

if __name__ == '__main__':
    test_cases = [
        {
            "question": "Please list the zip code of all the charter schools in Fresno County Office of Education.",
            "db_id": "california_schools"
        },
        {
            "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
            "db_id": "california_schools"
        }
    ]
    
    for test_case in test_cases:
        try:
            input_data = ClassifierInput(**test_case)
            result = classifier(input_data)
            print(f"\nQuestion: {test_case['question']}")
            print(f"SQL Components: {result}")
        except Exception as e:
            print(f"Error processing test case: {e}")
