import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from module import get_category, get_user_parser
import pandas as pd 

df = pd.read_csv('./pet_travel_chatbot_test_cases_bom.csv')

user_input = df['user_input'].tolist()
category_result = []
user_parser_result = []

# 테스트 케이스 실행 및 결과 저장 
for input_query in user_input:
    print(input_query)
    category = get_category(input_query)
    user_parser  = get_user_parser(input_query)
    print(category)
    print(user_parser)
    print('--------------------------------')
    category_result.append(category)
    user_parser_result.append(user_parser)

df['category_output'] = category_result
df['user_parser_output'] = user_parser_result

df.to_csv('./pet_travel_chatbot_test_cases_bom_result.csv', index=False)