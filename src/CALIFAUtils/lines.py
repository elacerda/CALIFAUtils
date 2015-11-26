#!/usr/bin/python
import numpy as np
'''
lines - the shape of the line function and the values used in the a,b and c
constants can be find at Cid et al. (2010) - Alternative diagnostic diagramas
and the 'forgotten' population of weak line galaxies in the SDSS.

Usage:
import lines

l = lines.Lines()
print l.lines

for t in l.lines:
    x = l.x[t]
    y = l.y[t]

    plot(x, y)

if you want to add a line do it like this:
def someFuncLine(c, x):
    a = c[0]
    b = c[1]
    return a * x + b

const = (1.0, 2.0)
x = np.linspace(-2.0, 2.0, 20)

l.addLine('MyLineType', someFuncLine, const, x)
'''
class Lines:
    def __init__(self, xn = 1000, create_BPT = True):
        self.xn = xn
        self.lines = []
        self.x = {}
        self.y = {}
        self.methods = {}
        self.consts = {}
        '''
        _var it's a convention to protect variables.Single underscore prefixed
        variables means that is a "private" attribute. Expect your users to 
        respect convention, and expect your abusers to break it no matter what 
        you do.
        '''
        self._removable = {} 
        
        if create_BPT is True:
            self.linesbpt = []
            self.linesbpt_init(self)

    def linesbpt_init(self, linename):
        x = {
            'K01' : np.linspace(-10.0, 0.47, self.xn + 1),
            'K03' : np.linspace(-10.0, 0.05, self.xn + 1),
            'S06' : np.linspace(-10.0, -0.2, self.xn + 1)
        }

        self.addLine('K01', self.linebpt, (1.19, 0.61, -0.47), x['K01'][:-1])
        self.addLine('K03', self.linebpt, (1.30, 0.61, -0.05), x['K03'][:-1])
        self.addLine('S06', self.linebpt, (0.96, 0.29, 0.2), x['S06'][:-1])

        x['K06'] = np.linspace(-10.0, 10.0, self.xn)
        self.addLine('K06', self.lline, (1.01, 0.48), x['K06'])
        self.fixK06(self)

        '''
        Just a shortcut to BPT lines.
        '''
        self.linesbpt = self.lines

        '''
        Blocking the lines to be removable by self.remLine(line)
        '''
        self.removable_init(self)

    def belowlinebpt(self, linename, x, y):
        if self.lines.__contains__(linename):
            mask = (y <= self.get_yfromx(linename, x)) 
            if linename != 'K06':
                c = self.consts[linename]
                mask &= (x < -1.0*c[-1])
        return mask

    @staticmethod
    def fixK06(self):
        yK06 = self.y['K06']
        yS06 = self.get_yfromx('S06', self.x['K06'])
        i = np.where(yK06 > yS06)[0][0]
        xmin = self.x['K06'][i]
        newx = np.linspace(xmin, 2.0, self.xn)
        self.remLine('K06')
        self.addLine('K06', self.lline, (1.01, 0.48), newx)

    @staticmethod
    def removable_init(self):
        self._removable = {
            'K01' : False,
            'K03' : False,
            'K06' : False,
            'S06' : False
        }

    def remLine(self, linename):
        if self.lines.__contains__(linename):
            if self._removable[linename]:
                self.lines.remove(linename)
                del self.methods[linename]
                del self.consts[linename]
                del self.x[linename]
                del self.y[linename]
            else:
                print 'line %s is not removable' % linename
        else:
            print 'line %s doesn\'t exist' % linename

    def addLine(self, linename, lfunc, lconst, x, removable = True):
        if not self.lines.__contains__(linename):
            self.addType(self, linename)
            self.addMethod(self, linename, lfunc)
            self.addConst(self, linename, lconst)
            
            self.x[linename] = x
            self.y[linename] = self.get_yfromx(linename, x)
            
            self._removable[linename] = removable
        else:
            print 'line %s exists, try another name' % linename

    @staticmethod
    def addType(self, linename):
        self.lines.append(linename)
    
    @staticmethod
    def remType(self, linename):
        self.lines.remove(linename)

    @staticmethod
    def addMethod(self, linename, lfunc):
        self.methods[linename] = lfunc

    @staticmethod
    def remMethod(self, linename):
        del self.methods[linename]

    @staticmethod
    def addConst(self, linename, lconst):
        self.consts[linename] = lconst
    
    @staticmethod
    def remConst(self, linename):
        del self.consts[linename]

    @staticmethod
    def linebpt(c, x):
        return c[0] + c[1] / (c[2] + x)
    
    @staticmethod
    def lline(c, x):
        return np.polyval(c, x)
        #return c[0] * x + c[1]

    def get_yfromx(self, linename, x):
        return self.methods[linename](self.consts[linename], np.array(x))

# vim: set et ts=4 sw=4 sts=4 tw=80 fdm=marker:
