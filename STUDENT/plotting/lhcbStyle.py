from ROOT import TStyle, gROOT, TLatex, TPaveText, TText

lhcbNames = []
lhcbStyle = None
lhcbLabel = None
lhcbLatex = None 

def setE5Style():
    style = TStyle("E5Style","Standard E5 plots style")

    font_id = 132
    basic_text_size = 0.06
    basic_fg_colour = 1
    basic_bg_colour= 0
    basic_line_width = 2.00

    style.SetTextFont(font_id)
    style.SetTextSize(basic_text_size)

    style.SetFrameBorderMode(0)
    style.SetFrameFillColor(basic_bg_colour)
    style.SetFrameLineWidth( int(basic_line_width) )    
    
    style.SetPaperSize(20,26)
    
    style.SetCanvasBorderMode(0)
    style.SetCanvasColor(basic_bg_colour)

    style.SetOptStat(0)  
    style.SetOptTitle(0) 
    style.SetOptFit(0)   
    style.SetOptDate(0)  

    style.SetPadBorderMode(0)
    style.SetPadColor(basic_bg_colour)
    style.SetPadTopMargin(0.07)
    style.SetPadRightMargin(0.10101010101010101010) 
    style.SetPadBottomMargin(0.16)
    style.SetPadLeftMargin(0.18)
    
    style.SetPadTickX(1) 
    style.SetPadTickY(1) 

    style.SetTickLength(0.03,"x")
    style.SetTickLength(0.03,"y")
    style.SetTickLength(0.03,"z")
    
    style.SetPadGridX(False) 
    style.SetPadGridY(False) 

    style.SetGridWidth(int(basic_line_width) )
    style.SetGridColor(basic_fg_colour)
    
    style.SetTitleStyle(0)
    style.SetTitleBorderSize(0)
    style.SetTitleTextColor(basic_fg_colour)
    style.SetTitleFillColor(basic_bg_colour)

    style.SetTitleFont(font_id,"title") 
    style.SetTitleFont(font_id,"x")
    style.SetTitleFont(font_id,"y")
    style.SetTitleFont(font_id,"z")

    style.SetTitleSize(1.2*basic_text_size,"x")
    style.SetTitleSize(1.2*basic_text_size,"y")
    style.SetTitleSize(1.2*basic_text_size,"z")

    style.SetTitleOffset(0.95,"x")
    style.SetTitleOffset(1.20,"y")
    style.SetTitleOffset(1.20,"z")

    style.SetTitleX(0.00)
    style.SetTitleY(1.00)

    style.SetTitleW(1.00)
    style.SetTitleH(0.05)    
    
    style.SetLabelFont(font_id,"x")
    style.SetLabelFont(font_id,"y")
    style.SetLabelFont(font_id,"z")

    style.SetLabelSize(basic_text_size,"x")
    style.SetLabelSize(basic_text_size,"y")
    style.SetLabelSize(basic_text_size,"z")

    style.SetLabelOffset(0.010,"X")
    style.SetLabelOffset(0.005,"Y")

    style.SetStatColor(0)
    style.SetStatBorderSize(0)
    style.SetStatFont(font_id)
    style.SetStatFontSize(0.05)
    style.SetStatFormat("6.3g")
    style.SetStatX(0.9)
    style.SetStatY(0.9)
    style.SetStatW(0.25)
    style.SetStatH(0.15)
    
    style.SetLegendBorderSize(0)
    
    style.SetFillColor(1)
    style.SetFillStyle(1001)
    
    style.SetPalette(1)
    
    style.SetHistLineColor(basic_fg_colour)
    style.SetHistLineWidth(int(basic_line_width) )
    style.SetHistLineStyle(0)

    #style.SetHistFillColor(basic_bg_colour)
    #style.SetHistFillStyle(0)

    style.SetHistMinimumZero(False)
    style.SetHistTopMargin(0.05)

    
    style.SetNdivisions(505,"x")
    style.SetNdivisions(510,"y")
        
    style.SetMarkerStyle(20)
    style.SetMarkerSize(1.0)
    style.SetMarkerColor(basic_fg_colour)

    style.SetErrorX(0.)      
    style.SetEndErrorSize(2) 

    style.SetFuncColor(basic_fg_colour)
    style.SetFuncWidth(int(basic_line_width) )
  
    gROOT.SetStyle("E5Style")
    gROOT.ForceStyle()
    return style



def setLHCbStyle():
    global lhcbStyle
    global lhcbText
    global lhcbLatex
    
    lhcbStyle = TStyle("lhcbStyle","Standard LHCb plots style")

    # use times new roman
    lhcbFont = 132
    # line thickness
    lhcbWidth = 2 
    lhcbTSize = 0.06

    #
    lhcbStyle.SetFrameBorderMode(0)
    lhcbStyle.SetCanvasBorderMode(0)
    lhcbStyle.SetPadBorderMode(0)
    lhcbStyle.SetPadColor(0)
    lhcbStyle.SetCanvasColor(0)
    lhcbStyle.SetStatColor(0)
    lhcbStyle.SetPalette(1)
    
    lhcbStyle.SetLegendBorderSize(0)
    lhcbStyle.SetLegendFont(132)
    lhcbStyle.SetFillColor(1)
    lhcbStyle.SetFillStyle(1001)
    
    # set the paper & margin sizes
    lhcbStyle.SetPaperSize(20,26)

    lhcbStyle.SetPadTopMargin(0.1)

    lhcbStyle.SetPadRightMargin(0.05) 
    lhcbStyle.SetPadBottomMargin(0.16)
    lhcbStyle.SetPadLeftMargin(0.14)
    
    # use large fonts
    lhcbStyle.SetTextFont(lhcbFont)
    lhcbStyle.SetTextSize(lhcbTSize)
    #  lhcbStyle.SetTextSize(0.08)
    lhcbStyle.SetLabelFont(lhcbFont,"x")
    lhcbStyle.SetLabelFont(lhcbFont,"y")
    lhcbStyle.SetLabelFont(lhcbFont,"z")
    lhcbStyle.SetLabelSize(lhcbTSize,"x")
    lhcbStyle.SetLabelSize(lhcbTSize,"y")
    lhcbStyle.SetLabelSize(lhcbTSize,"z")
    lhcbStyle.SetTitleFont(lhcbFont)
    lhcbStyle.SetTitleFont(lhcbFont,"x")
    lhcbStyle.SetTitleFont(lhcbFont,"y")
    lhcbStyle.SetTitleFont(lhcbFont,"z")
    lhcbStyle.SetTitleSize(1.2*lhcbTSize,"x")
    lhcbStyle.SetTitleSize(1.2*lhcbTSize,"y")
    lhcbStyle.SetTitleSize(1.2*lhcbTSize,"z")
    
    # use bold lines and markers
    lhcbStyle.SetLineWidth(lhcbWidth)
    lhcbStyle.SetFrameLineWidth(lhcbWidth)
    lhcbStyle.SetHistLineWidth(lhcbWidth)
    lhcbStyle.SetFuncWidth(lhcbWidth)
    lhcbStyle.SetGridWidth(lhcbWidth)
    lhcbStyle.SetLineStyleString(2,"[12 12]") 
    lhcbStyle.SetMarkerStyle(20)
    lhcbStyle.SetMarkerSize(1.0)
    
    # label offsets
    lhcbStyle.SetLabelOffset(0.010)


    #titles
    lhcbStyle.SetTitleOffset(0.95,"X")
    lhcbStyle.SetTitleOffset(0.95,"Y")
    lhcbStyle.SetTitleOffset(1.2,"Z")
    lhcbStyle.SetTitleFillColor(0)
    lhcbStyle.SetTitleStyle(0)
    lhcbStyle.SetTitleBorderSize(0)
    lhcbStyle.SetTitleFont(lhcbFont,"title")
    lhcbStyle.SetTitleX(0.0)
    lhcbStyle.SetTitleY(1.0) 
    lhcbStyle.SetTitleW(1.0)
    lhcbStyle.SetTitleH(0.05)
    
    # by default, do not display histogram decorations:
    lhcbStyle.SetOptStat(0)  
    #lhcbStyle.SetOptStat("emr")     # show only nent -e , mean - m , rms -r
    #lhcbStyle.SetStatFormat("6.3g") # specified as c printf options
    lhcbStyle.SetOptTitle(0)
    lhcbStyle.SetOptFit(0)
    #lhcbStyle.SetOptFit(1011) # order is probability, Chi2, errors, parameters

    # look of the statistics box:
    lhcbStyle.SetStatBorderSize(0)
    lhcbStyle.SetStatFont(lhcbFont)
    lhcbStyle.SetStatFontSize(0.05)
    lhcbStyle.SetStatX(0.9)
    lhcbStyle.SetStatY(0.9)
    lhcbStyle.SetStatW(0.25)
    lhcbStyle.SetStatH(0.15)

    # put tick marks on top and RHS of plots
    lhcbStyle.SetPadTickX(1)
    lhcbStyle.SetPadTickY(1)

    # histogram divisions: only 5 in x to avoid label overlaps
    lhcbStyle.SetNdivisions(505,"x")
    lhcbStyle.SetNdivisions(505,"y")
    lhcbStyle.SetNdivisions(505,"z")


    # define style for text
    lhcbLabel =  TText()
    lhcbLabel.SetTextFont(lhcbFont)
    lhcbLabel.SetTextColor(1)
    lhcbLabel.SetTextSize(0.04)
    lhcbLabel.SetTextAlign(12)

    # define style of latex text
    lhcbLatex = TLatex()
    lhcbLatex.SetTextFont(lhcbFont)
    lhcbLatex.SetTextColor(1)
    lhcbLatex.SetTextSize(0.04)
    lhcbLatex.SetTextAlign(12)

    # set this style
    gROOT.SetStyle("lhcbStyle")
    gROOT.ForceStyle()
    return lhcbStyle


def printLHCb( optLR = 'L', isPrelim = False , optText = ''):
    global lhcbStyle
    global lhcbNames
    
    lhcbName = None
    
    if ( optLR is 'R' ):
        lhcbName = TPaveText( 0.70 - lhcbStyle.GetPadRightMargin(),
                              0.85 - lhcbStyle.GetPadTopMargin(),
                              0.95 - lhcbStyle.GetPadRightMargin(),
                              0.95 - lhcbStyle.GetPadTopMargin(),
                              "BRNDC" )
    elif ( optLR is 'L' ):
        lhcbName = TPaveText(lhcbStyle.GetPadLeftMargin() + 0.05,
                             0.85 - lhcbStyle.GetPadTopMargin(),
                             lhcbStyle.GetPadLeftMargin() + 0.30,
                             0.95 - lhcbStyle.GetPadTopMargin(),
                             "BRNDC")
    elif ( optLR is 'BR' ):
        lhcbName = TPaveText( 0.70 - lhcbStyle.GetPadRightMargin(),
                              0.05 + lhcbStyle.GetPadBottomMargin(),
                              0.95 - lhcbStyle.GetPadRightMargin(),
                              0.15 + lhcbStyle.GetPadBottomMargin(),
                              "BRNDC" )

    if ( isPrelim  ):
        lhcbName.AddText('#splitline{LHCb}{#scale[1.0]{Preliminary}}')
    else:
        lhcbName.AddText('LHCb')
        
    
    lhcbName.SetFillColor(0)
    lhcbName.SetTextAlign(12)
    lhcbName.SetBorderSize(0)
    lhcbName.Draw()


    lhcbNames += [ lhcbName ]
    return 

