from farm.infer import QAInferencer
from farm.data_handler.inputs import QAInput, Question

nlp = QAInferencer.load(
    "deepset/roberta-base-squad2",
    task_type="question_answering",
    batch_size=16,
    num_processes=0)

input = QAInput(
    doc_text="My name is Lucas and I live on Mars.",
    questions=Question(text="Who lives on Mars?",
                       uid="your-id"))

res = nlp.inference_from_objects([input], return_json=False)[0]

# High level attributes for your query
print(res.question)
print(res.context)
print(res.no_answer_gap)
# ...
# Attributes for individual predictions (= answers)
pred = res.prediction[0]
print(pred.answer)
print(pred.answer_type)
print(pred.answer_support)
print(pred.offset_answer_start)
print(pred.offset_answer_end)