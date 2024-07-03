from synthetica.decorators import callback


class Model:
    def __init__(self, param):
        self._param = param
        
    @property
    def param(self) -> float:
        """Mean value"""
        return self._param

    @param.setter
    @callback('func')
    def param(self, value) -> float:
        """Mean value update"""
        if value != self._param:
            self._param = value
    
    @property
    def func(self):
        return self.param + 1
        

def test_callback():
    model = Model(1)
    assert model.param == 1
    assert model.func == 2
    
    model.param = 2
    assert model.param == 2
    assert model.func == 3
    
    
    
    