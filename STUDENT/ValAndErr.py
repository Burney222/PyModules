from math import sqrt

class ValAndErr(object):
    """ Class for primitive and therefore fast propagation of uncertainties.
        Works only for standard operations and does not calculate derivatives.
    """

    @staticmethod
    def check_other_value(other_value):
        if type(other_value) == ValAndErr:
            return other_value
        else:
            return ValAndErr(other_value)

    def __init__(self, _v, _e=0 ) :
        if type(_v) == ValAndErr:
            self.v = _v.v
            self.e = _v.e
        else:  
            self.v = float(_v)
            self.e = float(_e)

    def __mul__(self, other_value ) :
        other_value = ValAndErr.check_other_value(other_value)
        va = self.v
        vb = other_value.v
        ea = self.e
        eb = other_value.e
        tmpv = va * vb
        tmpe = sqrt( ea*ea*vb*vb + eb*eb*va*va ) 
        return ValAndErr(tmpv, tmpe)

    def __div__(self, other_value ) :
        other_value = ValAndErr.check_other_value(other_value)
        va = self.v
        vb = other_value.v
        ea = self.e
        eb = other_value.e
        tmpv = va/vb
        tmpe = sqrt( ea*ea/(vb*vb) + eb*eb*va*va/(vb*vb*vb*vb) )
        return ValAndErr(tmpv, tmpe)

    def __add__(self, other_value ) :
        other_value = ValAndErr.check_other_value(other_value)
        va = self.v
        vb = other_value.v
        ea = self.e
        eb = other_value.e
        tmpv = va+vb
        tmpe = sqrt( ea*ea + eb*eb ) 
        return ValAndErr(tmpv, tmpe)

    def __sub__(self, other_value ) :
        other_value = ValAndErr.check_other_value(other_value)
        va = self.v
        vb = other_value.v
        ea = self.e
        eb = other_value.e
        tmpv = va-vb
        tmpe = sqrt( ea*ea + eb*eb ) 
        return ValAndErr(tmpv, tmpe)


    def __rmul__(self, other_value ) :
        return self.__mul__(other_value)

    def __rdiv__(self, other_value ) :
        other_value = ValAndErr.check_other_value(other_value)
        return other_value.__div__(ValAndErr(self.v, self.e))

    def __radd__(self, other_value ) :
        return self.__add__(other_value)

    def __rsub__(self, other_value ) :
        return (self*-1.).__add__(other_value)

    def __str__(self) :
        return "{0} +/- {1}".format(self.v, self.e)

    __repr__ = __str__

    def __iadd__(self, other_value):
        tmpval = ValAndErr( self.v , self.e ).__add__( other_value)
        self.v = tmpval.v
        self.e = tmpval.e
        return self

    def __isub__(self, other_value):
        tmpval = ValAndErr( self.v , self.e ).__sub__(other_value)
        self.v = tmpval.v
        self.e = tmpval.e
        return self

    def __imul__(self, other_value):
        tmpval = ValAndErr( self.v , self.e ).__mul__(other_value)
        self.v = tmpval.v
        self.e = tmpval.e
        return self

    def __idiv__(self, other_value):
        tmpval = ValAndErr( self.v , self.e ).__div__(other_value)
        self.v = tmpval.v
        self.e = tmpval.e
        return self
    
    def get_relative_error(self) :
        return self.e / self.v

    def get_error(self) :
        return self.e 

    def get_value(self) :
        return self.v

    def sqrt(self):
        relErr = self.get_relative_error()
        return ValAndErr( sqrt(self.v), 0.5*relErr*sqrt(self.v) )
