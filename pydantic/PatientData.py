from pydantic import BaseModel, EmailStr
from typing import List, Dict,Optional

class Patient(BaseModel):
    name: str
    email: EmailStr # only valid email is accepted
    age: int
    weight: float
    married: bool = False # bydefult false is assigned to married
    allergies: Optional[List[str]] = None, # it will only accept list of str
    contact_detials: Dict[str, str] # it will only accept dictionary in contact_detials and that dictionary must contain string value

def patient_data(patient:Patient):
    print(patient.name)
    print(patient.married)
    print(patient.contact_detials)

def patient_update(patient: Patient):
    print(patient.name)
    print(patient.age)
    print(patient.allergies)
    print("Updated")


patient_info = {
    'name': 'Priyanshu yadav',
    'age': 21,
    'weight':65.4,
    'married': False,
    'contact_detials': {'gmail':'abc@gmail.com', 'phone':'32124354324'}
}
patient1 = Patient(**patient_info)
patient_update(patient1)
