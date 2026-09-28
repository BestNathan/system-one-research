from __future__ import annotations
import os, time
from typing import Any, Protocol

class SystemOneBackend(Protocol):
    name: str
    def decide(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], float]: ...

class LayaBackend:
    name = "laya"
    def __init__(self):
        import laya
        model = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
        device = os.getenv("LAYA_DEVICE", "cpu")
        t=time.perf_counter()
        self.agent=laya.load(model, device=device)
        self.load_ms=(time.perf_counter()-t)*1000
        self.model=model
    def decide(self,state,questions):
        t=time.perf_counter()
        result=self.agent.predict(state,questions)
        return result,(time.perf_counter()-t)*1000

class JevBackend:
    name = "jev"
    def __init__(self):
        import httpx
        key=os.environ["TYPESAFE_API_KEY"]
        self.model=os.getenv("JEV_MODEL","jev-1.13.0")
        self.client=httpx.Client(base_url="https://api.typesafe.ai",timeout=90,headers={"Authorization":f"Bearer {key}"})
        self.load_ms=0.0
    def decide(self,state,questions):
        t=time.perf_counter()
        r=self.client.post("/v1/systemone",json={"model":self.model,"state":state,"questions":questions})
        elapsed=(time.perf_counter()-t)*1000
        r.raise_for_status()
        return r.json(),elapsed

def create_backend(name:str)->SystemOneBackend:
    return LayaBackend() if name=="laya" else JevBackend()
