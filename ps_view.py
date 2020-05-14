from psfile import PSFile

from random import uniform
from psfile import EPSFile
from psfile import PSFile

fd = EPSFile("ex1.eps", 600, 600)

# dark gray background
fd.append("0.1 setgray")
fd.append("0 0 %d %d rectfill"%(fd.width, fd.height))

# a grid of dark orange lines
fd.append("1 .596 .118 setrgbcolor")
fd.append("1 setlinewidth")
for i in range(1,5):
    y = 100*i/5.0
    fd.append("5 %.1f moveto 595 %.1f lineto"%(y,y))
for i in range(1,30):
    x = 100*i/5.0
    fd.append("%.1f 5 moveto %.1f 95 lineto"%(x,x))
fd.append("stroke")



fd.close()