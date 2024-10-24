import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

text = """
[이투뉴스] 지난해 국내 수소차 판매량이 2022년 대비 55.2% 감소하며 중국에게 판매량 1위 자리를 내줬다. 한정된 차량 선택지와 충전 인프라 부족 문제가 해결되지 못하며 국내서 수소차 입지가 줄고 있다.

SNE리서치에 따르면 지난해 전 세계에서 판매된 수소차 판매량은 1만4451대로 집계됐다. 이는 2022년과 비교해 30.2% 감소한 양이다. 

업체별로는 지난해 현대자동차는 넥쏘와 일렉시티를 포함해 5012대 판매하며 시장 점유율 34.7%로 시장 선두 자리를 유지했다. 다만 2022년 넥쏘 판매량 1만1179대에서 4709대로 크게 줄었다. 전년동기 55.9%나 감소했다. 

반면 도요타는 미라이를 3737대 판매했다. 판매량이 크게 감소한 현대차와 다르게 2022년보다 3.9% 늘었다. 

넥쏘 판매 부진 영향으로 지난해 우리나라 수소차 판매량도 2022년 1만336대보다 55.2% 감소한 4361대로 집계됐다.

중국은 우리나라의 판매량 감소와 반대로 상용차를 중심으로 지속적인 성장세를 보여 점유율 1위에 올랐다. 전기차에 이어 수소차도 가장 많이 판매하며 친환경차 시장서 강세를 보이고 있다.

SNE리서치는 “2018년 넥쏘 출시 이후 현재까지 국내 소비자가 선택할 수 있는 수소차는 넥쏘로 한정돼 있다”며 “수소 충전 비용 상승, 불량 사고, 충전 인프라 부족 등으로 국내 친환경차 시장에서 수소차의 매력은 떨어질 수밖에 없다”고 지적했다. 

반면 “중국은 수소에너지 산업 중장기 발전 계획을 통해 수소차 보급 확대 및 인프라 구축에 적극나서며 상업화를 진행하고 있다. 특히 상용차를 활용해 시장 점유율을 빠르게 높였다”고 설명했다.

국내 시장에서 수소차 판매량이 크게 감소했으나 여러 지자체가 수소차 구매 지원 계획을 잇달아 발표하고 있다. 각 지자체는 국비와 시비를 더해 3250~3750만원의 보조금을 지원한다.

다만 지원되는 차종은 여전히 현대차의 넥쏘뿐이다. 이에 일각에선 "국내 수소모빌리티 지원정책이 현대차에 한정돼선 안된다. 소비자 선택의 폭을 넓혀야 수소차 경쟁력이 살아날 수 있다"는 지적이 나오고 있다.

한편 지난해 크게 주춤한 수소차 시장에서 기업들은 신차 출시를 앞두고 있다. 도요타는 지난해 11월 이미 크라운 세단을 기반으로 한 수소차를 출시해 102대를 판매했다. 크라운에는 미라이와 동일한 연료전지시스템이 사용됐다.

혼다는 올해 중 일본과 북미 시장에 스포츠유틸리티차(SUV) CR-V를 기반으로 한 수소차 출시를 계획 중이다. 혼다는 2021년 수소차 클래리티를 단종하며 시장에서 철수했으나 다시 경쟁에 뛰어드는 모양새다. 현대차는 내년에 들어서야 넥쏘 신형을 출시할 예정이다.

출처 : 이투뉴스(http://www.e2news.com)
"""

model_name = './training_results/'
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == 'cuda':
    print(f'Server is running on GPU')
else:
    print(f'Server is running on CPU')

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
model = AutoModelForSequenceClassification.from_pretrained(model_name, problem_type = "single_label_classification")

model.eval()
model.to(device)

input_max_length=2048
model = model
tokenizer = tokenizer
device = device
label2sentiment = {0 : 'Negative', 1 : 'Positive'}
device=device

text_rmv = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', text)
text_rmv = text_rmv.replace('<br','').replace('>','')
     
input_encoding = tokenizer([text], truncation=True, padding=True, return_tensors='pt')
input_encoding = {key: torch.tensor(val).to(device) for key, val in input_encoding.items()}

with torch.no_grad():
    predictions = model(**input_encoding)
    print(predictions)
    predictions = predictions.logits.squeeze().argmax().item()

sentiment = label2sentiment[predictions]



