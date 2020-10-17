"""

Agentpy
Toolbox for other modules

Copyright (c) 2020 Joël Foramitti

"""

class AgentpyError(Exception):
    pass



class attr_dict(dict):
    
    """ Dictionary where attributes and dict entries are identical. """
    
    # By Kimvais https://stackoverflow.com/a/14620633/
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.__dict__ = self 
    
    def __repr__(self):
        return f"<attr_dict>" 
        
   
            
def make_list(element,keep_none=False):
    
    """ Turns `element` into a list of itself if it is not of type list or tuple. """
    
    if element is None and not keep_none: element = [] # Convert none to empty list
    if not isinstance(element, (list, tuple)): element = [element]
        
    return element



class func_list(list):
    
    """ List that distributes calls to its members and returns list of return values """
    
    def __init__(self,_super,_name,*args):
        super().__init__(*args)
        self._super = _super
        self._name = _name
    
    def __call__(self,*args,**kwargs):
        
        try: return [ func_obj(*args,**kwargs) for func_obj in self ]
        except TypeError: raise TypeError(f"Not all objects in '{type(self._super).__name__}' are callable.")
    
    def __eq__(self, other):
        
        return type(self._super)([obj for obj,x in zip(self._super,self) if x == other])
    
    def __ne__(self, other):
        
        return type(self._super)([obj for obj,x in zip(self._super,self) if x != other])

    def __lt__(self, other):
        
        return type(self._super)([obj for obj,x in zip(self._super,self) if x < other])
    
    def __le__(self, other):
        
        return type(self._super)([obj for obj,x in zip(self._super,self) if x <= other])
    
    def __gt__(self, other):
        
        return type(self._super)([obj for obj,x in zip(self._super,self) if x >= other])
    
    def __ge__(self, other):
        
        return type(self._super)([obj for obj,x in zip(self._super,self) if x > other])
    
    def __add__(self,v):    
        return func_list(self._super,self._name,[x+v for x in self])
    
    def __sub__(self,v):    
        return func_list(self._super,self._name,[x-v for x in self])
    
    def __mul__(self,v):    
        return func_list(self._super,self._name,[x*v for x in self])
    
    def __truediv__(self,v):    
        return func_list(self._super,self._name,[x/v for x in self])
    
    def __iadd__(self,v):
        return self + v
    
    def __isub__(self,v):
        return self - v
    
    def __imul__(self,v):
        return self * v
    
    def __itruediv__(self,v):
        return self / v
    
class attr_list(list):
    
    """ A list that can access and assign attributes of it's entries like it's own """
        
    def __setattr__(self, name, value):
        
        if isinstance(value,func_list):
            for obj,v in zip(self,value):
                setattr(obj, name, v)
            return
        
        for obj in self:
            setattr(obj, name, value)
    
    def __getattr__(self,name):
        
        try: 
            return func_list( self, name, [ getattr(obj,name) for obj in self ] )
        except AttributeError:
            raise AttributeError(f"Neither '{type(self).__name__}' object nor it's entries have attribute '{name}'")