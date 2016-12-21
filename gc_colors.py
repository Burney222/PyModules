import matplotlib.colors as colors
import pprint

gc_colors = { "gcred" : "#BE1818",
              "gcorange" : "#FF9900",
              "gclime" : "#9DCE09",
              "gcgreen" : "#488F38",
              "gcdarkgrey" : "#808080",
              "gcgrey" : "#C0C0C0",
              "gclightgrey" : "#E6E6E6",
              "gcsilver" : "#EFF2F9",
              "gcdarkblue" : "#00006E",
              "gcblue" : "#000099",
              "gcgreyblue" : "#1C3363",
              "gccyan" : "#009999",
              "gcgreycyan" : "#BBE0E3" }


#Patch matplotlib color palette
colors.cnames.update(gc_colors)


def show():
    #Function to plot the GC colors
    pprint.pprint(gc_colors)
