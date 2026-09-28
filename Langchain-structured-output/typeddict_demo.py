from typing import TypedDict

class Person(TypedDict):

    name: str
    age: int

new_person: Person={'name':'Priyanshu yadav','age':21}
print(new_person)