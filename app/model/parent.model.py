from pydantic import Basemodel

class ParentCunk(Basemodel):
  parent_id: str
  text: str