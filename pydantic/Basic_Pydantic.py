from pydantic import BaseModel

class Patient(BaseModel):
    name: str
    age: int
    weight: float

def insert_patient_data(patient:Patient):
    print(patient.name)
    print(patient.age)
    print(patient.weight)
    print("insert")

def update_patient_data(patient:Patient):
    print(patient.name)
    print(patient.age)
    print("update")

patient_info = {'name':'priyanshu yadav','age':21,'weight':65} #. if '65' written in string pydanric automatically convert to int
patient1 = Patient(**patient_info)
insert_patient_data(patient1)

