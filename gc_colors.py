import matplotlib as mpl
import pprint

gc_colors = { u"gcred" : u"#BE1818",
              u"gcorange" : u"#FF9900",
              u"gclime" : u"#9DCE09",
              u"gcgreen" : u"#488F38",
              u"gcdarkgrey" : u"#808080",
              u"gcgrey" : u"#C0C0C0",
              u"gclightgrey" : u"#E6E6E6",
              u"gcsilver" : u"#EFF2F9",
              u"gcdarkblue" : u"#00006E",
              u"gcblue" : u"#000099",
              u"gcgreyblue" : u"#1C3363",
              u"gccyan" : u"#009999",
              u"gcgreycyan" : u"#BBE0E3" }


#Patch matplotlib color palette

if int(mpl.__version__[0]) < 2: 
    mpl.colors.cnames.update(gc_colors)
else:
    mpl.colors._colors_full_map.update(gc_colors)

def show():
    #Function to plot the GC colors
    pprint.pprint(gc_colors)
