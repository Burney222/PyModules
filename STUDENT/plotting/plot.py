def plot_correlations(data,label='', **kwds):
    """Calculate pairwise correlation between features.
    
    Arguemnts:
    data: Pandas.DataFrame on which the correlations are calculated.

    label: prefix for the plot title and file name.

    Extra arguments are passed on to DataFrame.corr()
    """
    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds)

    fig, ax1 = plt.subplots(ncols=1, figsize=(10,9))
    
    opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
    heatmap1 = ax1.pcolor(corrmat, **opts)
    fig.colorbar(heatmap1, ax=ax1)

    ax1.set_title(label+" Correlations_{}vars".format(len(data.columns)-1))

    labels = corrmat.columns.values
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, fontsize=12,minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, fontsize=12,minor=False)
        
    fig.set_tight_layout(True)
    fig.savefig(label+'_correlation_{}vars.pdf'.format(len(data.columns)))
    fig.savefig(label+'_correlation_{}vars.png'.format(len(data.columns)))

#==================================
#==================================

def _color_coded_pull_hists(pull_hist):
	"""Helper function for 'plot_with_pulls'.
	"""
	import ROOT as R
	pulls1 = pull_hist.Clone("pulls1")
	pulls2 = pull_hist.Clone("pulls2")
	pulls3 = pull_hist.Clone("pulls3")
	pulls4 = pull_hist.Clone("pulls4")

	#Additional pull histograms for color coding only used if normalize = true
	pulls1.SetFillColor(18)
	pulls2.SetFillColor(16)
	pulls3.SetFillColor(14)
	pulls4.SetFillColor(12)

	pulls1.SetLineWidth(2)
	pulls2.SetLineWidth(2)
	pulls3.SetLineWidth(2)
	pulls4.SetLineWidth(2)

	for i in range(pull_hist.GetNbinsX()):
		pulls1.SetBinContent(i,0)
		pulls2.SetBinContent(i,0)
		pulls3.SetBinContent(i,0)
		pulls4.SetBinContent(i,0)
	
		if R.TMath.Abs(pull_hist.GetBinContent(i)) < 1:
			pulls1.SetBinContent(i,pull_hist.GetBinContent(i))
		elif R.TMath.Abs(pull_hist.GetBinContent(i)) < 2:
			pulls2.SetBinContent(i,pull_hist.GetBinContent(i))
		elif R.TMath.Abs(pull_hist.GetBinContent(i)) < 3:
			pulls3.SetBinContent(i,pull_hist.GetBinContent(i))
		else:
			pulls4.SetBinContent(i,pull_hist.GetBinContent(i))

	return (pulls1, pulls2, pulls3, pulls4)

#==================================
#==================================

def _RunTest(hist):
	"""Helper function for 'plot_with_pulls'.
	"""
	import ROOT as R
	runs = 0
	entries = hist.GetNbinsX()
	nplus = 0
	nminus = 0
	sign = 1
	pval = 0

	for i in range(entries):
		pval = hist.GetBinContent(i)
    # only count pulls different from zero (due to numerics != 0.0 won't work here)
		if (R.TMath.Abs(pval) > 1e-4):
			if float(sign) * pval < 0:
				runs+=1
			if pval > 0:
				nplus+=1
				sign = 1
			elif pval <= 0:
				nminus+=1
				sign = -1

	n = nplus+nminus
	if n <1:
		print("Warning: n in RunTest is {0}".format(n))
		return float('nan')
	expect = 2.*nplus*nminus/(n) + 1.
	variance = 2.*nplus*nminus*(2.*nplus*nminus-n)/(n*n*(n-1.))
	if variance > 0:
		return (R.TMath.Erf((runs-expect)/R.TMath.Sqrt(variance))+1.)/2.
	else:
		print("Warning: variance in RunTest is {0}".format(variance))
		return float('nan')



def _pull_distribution(pull_hist_raw, chi2_reduced, chi2_pvalue, name):
	"""Helper function for 'plot_with_pulls'.
	"""
	from array import array
	import ROOT as R
	h_pulls = R.TH1D("hGauss","hGauss;Pull [#sigma];Number of bins",10,-5,5)

	p_can =  R.TCanvas("p_can", "", 800, 600)

	num_pulls_used = 0

	for i in range(pull_hist_raw.GetNbinsX()):
		if R.TMath.Abs(pull_hist_raw.GetBinContent(i)) > 1e-4:
			h_pulls.Fill(pull_hist_raw.GetBinContent(i))
			#print("Filling pull histogram with {0}".format(pull_hist_raw.GetBinContent(i)))
			num_pulls_used+=1

	if num_pulls_used < 1:
		print("Warning: no pull used in gauss fit!")
  
	f_gauss_norm = R.TF1("fGauss","gaus(0)/([2]*sqrt(2*3.1415))",-5,5)
	f_gauss_norm.SetParameter(0,num_pulls_used)
	f_gauss_norm.SetParameter(1,0)
	f_gauss_norm.SetParameter(2,1)
	f_gauss_norm.SetLineColor(R.kBlack)
	f_gauss_norm.SetLineWidth(4)

	f_gauss_fit = R.TF1("fFit","gaus(0)/([2]*sqrt(2*3.1415))",-5,5)
	f_gauss_fit.SetParameter(0,num_pulls_used)
	f_gauss_fit.SetParameter(1,0)
	f_gauss_fit.SetParameter(2,1)
	f_gauss_fit.SetLineColor(12)
	f_gauss_fit.SetLineStyle(R.kDashed)
	f_gauss_fit.SetLineWidth(4)

	h_pulls.SetFillColor(16)
	h_pulls.SetFillStyle(1001)
	h_pulls.SetLineColor(R.kBlack)
	h_pulls.SetLineWidth(2)


	f_gauss_norm.Draw()

	#h_pulls.Fit("fFit", "Q")
	h_pulls.Fit(f_gauss_fit, "Q")
	#Get error Band Histogram
	h_error = R.TH1D("hError","hError",500,-5,5)

	for i in range(h_error.GetNbinsX()):
		pos = array("d", [0])
		pos[0] = -5. + float(i)*10./500.+(10./500./2)

		dev0 = f_gauss_fit.GradientPar(0, pos, 0.01)
		dev1 = f_gauss_fit.GradientPar(1, pos, 0.01)
		dev2 = f_gauss_fit.GradientPar(2, pos, 0.01)

		err0 = f_gauss_fit.GetParError(0)
		err1 = f_gauss_fit.GetParError(1)
		err2 = f_gauss_fit.GetParError(2)

		error = R.TMath.Sqrt(dev0*dev0*err0*err0+dev1*dev1*err1*err1+dev2*dev2*err2*err2)

		h_error.SetBinContent(i,f_gauss_fit.Eval(pos[0]))
		h_error.SetBinError(i,error)


	h_error.SetFillColor(12)
	h_error.SetFillStyle(3001)
	h_error.SetLineColor(12)
	h_error.SetLineStyle(R.kDashed)
	h_error.SetLineWidth(4)
	h_error.SetMarkerStyle(0)

	h_pulls.SetMaximum( (h_pulls.GetMaximum()+4*R.TMath.Sqrt(h_pulls.GetMaximum()))*1.1 ) 

	#legend

	nVal = num_pulls_used - f_gauss_fit.GetParameter(0)
	mVal = f_gauss_fit.GetParameter(1)
	sVal = f_gauss_fit.GetParameter(2)-1

	nErr = f_gauss_fit.GetParError(0)
	mErr = f_gauss_fit.GetParError(1)
	sErr = f_gauss_fit.GetParError(2)

	legend = R.TLegend(0.2,0.65,0.48,0.93)
	legend.SetTextFont(132)
	legend.SetTextSize(0.04)
	header1 = "Run test: #it{p} = " + "{0:.2f}".format(_RunTest(pull_hist_raw))
	header2 = ", #chi^{2}/ndf = " + "{0:.2f} (p = {1:.2f})".format(chi2_reduced, chi2_pvalue)
	legend.SetHeader(header1 + header2)
	entry1 = "#splitline{Gaussian fit}{#scale[0.7]{#Delta#it{N}=" + "{0:.0f}#pm {1:.0f}, #Delta#mu={2:.2f}#pm {3:.2f}, #Delta#sigma={4:.2f}#pm {5:.2f}".format(nVal, nErr, mVal, mErr, sVal, sErr) + "}}"
	legend.AddEntry(h_error, entry1,"lf")
	legend.AddEntry(f_gauss_norm,"Normal distribution","l")
	legend.AddEntry(h_pulls,"Pull distribution","f")
	legend.SetFillColor(0)

	#redraw
	h_pulls.Draw()

	legend.Draw("NCP")

	h_pulls.Draw("same")
	h_error.Draw("E4 same")
	f_gauss_norm.Draw("same")

	R.gPad.RedrawAxis()
	p_can.SaveAs(name + "_pulldistribution.pdf")
	#return ((f_gauss_norm, ""), (h_pulls, ""), (h_pulls, "same"), (h_error, "E4 same"), (f_gauss_norm, "SAME"), (legend, "NCP"))

#==================================
#==================================

def _GetPulls(pFrame, normalize = True):
	"""Helper function for 'plot_with_pulls'.
	"""
	import ROOT as R
	from array import array
	
	data = None
	curve = None

	for i in range(int(pFrame.numItems())): #pFrame.numItems() returns double... dont ask!
		print("Checking item number {0} with name {1}".format(i, pFrame.getObject(i).GetName()))
		if pFrame.getObject(i).GetName().startswith("h_"): #RooDataSet is converted to a RooHist and gets h_ added at the beginning of its name
			print("Found data called {0}".format(pFrame.getObject(i).GetName()))
			data = pFrame.getObject(i)
			continue
		if "_Comp[" in pFrame.getObject(i).GetName(): # this is just one component of the full PDF
			continue
		print("Found curve called {0}".format(pFrame.getObject(i).GetName()))
		curve = pFrame.getObject(i)

	#print(curve.GetName())
	#curve.Print("V")
	#data = pFrame.getHist("h_data_jpsik_5050")

	if not curve or not data:
		if not curve:
			print("Error in during plotting. _GetPulls(...): Could not get curve!")
		if not data:
			print("Error in during plotting. _GetPulls(...): Could not get data!")

		pulls = R.TH1D("pulls","Pulls",100,0,1)
		return pulls

	limits = array("d", [])
	values = array("d", [])
	errors = array("d", [])

	x = array("d", [0])
	y = array("d", [0])
	e = 0
	c = 0

	data.GetPoint(0,x,y)

	limits.append(x[0]-data.GetErrorXlow(1))

	for i in range(data.GetN()):

		data.GetPoint(i,x,y);
		c = curve.Eval(x[0]);

		# use upper error if point lies beneath curve
		if (y[0] > c):
			e = data.GetErrorYlow(i);
		else:
			e = data.GetErrorYhigh(i);

		#fetch roofit fuckup
		if ( (R.TMath.Abs(y[0]) > 10000000000)):
		   	y[0] = c
		if ((e == 0) or (e  != e) or (R.TMath.Abs(e) > 10000000000)):
			e = 1
	#    if (c < 0) {
	#      // forcing PDF positive definite
	#      // commented out by Uli 12.06.2013, because Pulls for Asymmetry Plots are broken if this line is included
	#      c = 0;
	#    }
		if x[0] > pFrame.GetXaxis().GetXmax():
			limits[i]=pFrame.GetXaxis().GetXmax()
			break
	    # for zero bins and small pulls, rather set pulls to zero
	    # if not, this will result in unrealistic small pulls in sidebands and high
	    # lifetimes
		if y[0] == 0 and c < 0.5:
			c = 0

		#pulls
		if (normalize):
			limits.append(x[0]+data.GetErrorXhigh(i))
			values.append((y[0]-c)/e)
			errors.append(0)

		#residuals
		else:
			limits.append(x[0]-data.GetErrorXhigh(i))
			values.append((y[0]-c))
			errors.append(e)

	pulls = R.TH1D("pulls","Pulls",len(values),limits)

	for i in range(len(values)):
		pulls.SetBinContent(i,values[i-1])
		pulls.SetBinError(i,errors[i-1])

	return pulls


#==================================
#==================================


def plot_with_pulls(plot_frame, 
										plot_variable, 
										name, 
										output_path = "Plots/",
										num_fit_params = 0,
										label="",
										lab_pos = 'R',
										y_min_log = 0.11,
										legend=None,
										do_pulls=True,
										additional_objects = {}):
	""" Plots a variable with pulls relative to a fit function.

	Arguments: 
	plot_frame: RooPlot object containing data points and fit projections.

	plot_variable: RooRealVar observables related to the x axis.

	name: File name of the resulting pdf file containing fit in normal/log scale with/without pulls.

	output_path: Path to output file. 

	num_fit_params: Number of floating parameters in the fit.

	label: The usual label in form of a string like LHCb or LHCb preliminary (optional)

	lab_pos: Position where the label is printed, left (L), right (R) or bottom right (BR) (optional)
					 Can also be a list containing the coordinate in the order [X1,Y1,X2,Y2]

	y_min_log : lower border of the y axis when plotting in log scale (optional)

	legend : dictionary containing information on the legend (optional)
					 use "from collections import OrderedDict" to control the order of the elements in the legend
					 must contain "position" : (X1,Y1,X2,Y2)
					 and the for each line in the legend the name : (label, draw_option)
					 where the name is the one used in object.plotOn(my_frame, my_component, ..., RooFit.Name(name) )
					 if the object tis not present in the RooPlot but in additional_objects, the key is the object itself
					 eg object : (label, draw_option)

	do_pulls : create plots with pulls and a pull destribution (default is True)

	additional_objects : dictionary that can contain any drawable ROOT object alog with the desired draw option
											 e.g. additional_objects = { my_TLine : "same", my_TBox : "same"}
	"""
	import ROOT as R

	from pythonTools import setE5Style, ensure_dir
	stlyle = setE5Style()

	if output_path != "":
		ensure_dir(output_path)

	if name.endswith(".pdf"):
		name = name[:-4]

	plot_frame.SetMaximum(1.001*plot_frame.GetMaximum())
	#plot_frame.SetMinimum(1e-5) # remove empty bins from plot
	canvas = R.TCanvas("canvas{0}".format(name),"canvas",900,600)


	if legend:
		if "position" in legend and len(legend["position"]):
			pass
		else:
			print("legend must be a dictionary containing a key named position with a 4 element long list defining the position [X1,Y1,X2,Y2]")
			raise KeyError('Missing entry with key <position> in dictionary to build legend')


	#plot_frame.Draw()
	leg = None
	if legend:
		leg = R.TLegend(legend["position"][0] - stlyle.GetPadRightMargin(),
									legend["position"][1] - stlyle.GetPadTopMargin(),
									legend["position"][2] - stlyle.GetPadRightMargin(),
									legend["position"][3] - stlyle.GetPadTopMargin ()
									)

		leg.SetTextFont(132)
		leg.SetTextSize(0.055)
		leg.SetFillColor(0)
		leg.SetTextAlign(12)
		leg.SetBorderSize(0)
		


	lhcbName = None
	if label != "":
		if hasattr(lab_pos, "__len__"):# user defined position
			if not len(lab_pos) == 4:
				print("lab_pos must either be a string with one of the valid predifined positions or a list with [X1,Y1,X2,Y2]")
				raise ValueError("lab_pos must be string or list with len(lab_pos)==4")
			lhcbName = R.TPaveText( lab_pos[0] - stlyle.GetPadRightMargin(),
															lab_pos[1] - stlyle.GetPadTopMargin(),
															lab_pos[2] - stlyle.GetPadRightMargin(),
															lab_pos[3] - stlyle.GetPadTopMargin(),
															"BRNDC" )
		if ( lab_pos is 'R' ):
			lhcbName = R.TPaveText( 0.70 - stlyle.GetPadRightMargin(),
															0.85 - stlyle.GetPadTopMargin(),
															0.95 - stlyle.GetPadRightMargin(),
															0.95 - stlyle.GetPadTopMargin(),
															"BRNDC" )

		elif ( lab_pos is 'L' ):
			lhcbName = R.TPaveText( stlyle.GetPadLeftMargin() + 0.05,
														0.85 - stlyle.GetPadTopMargin(),
														stlyle.GetPadLeftMargin() + 0.30,
														0.95 - stlyle.GetPadTopMargin(),
														"BRNDC")
		elif ( lab_pos is 'BR' ):
			lhcbName = R.TPaveText( 0.70 - stlyle.GetPadRightMargin(),
														0.05 + stlyle.GetPadBottomMargin(),
														0.95 - stlyle.GetPadRightMargin(),
														0.15 + stlyle.GetPadBottomMargin(),
														"BRNDC" )

		lhcbName.AddText(label)
		lhcbName.SetFillColor(0)
		lhcbName.SetTextAlign(12)
		lhcbName.SetBorderSize(0)



	if legend:
		legend.pop("position", None)
		for comp, (lab, opt) in legend.items():
			if type(comp) is str:
				print("Adding {} to legend".format(lab))
				leg.AddEntry(plot_frame.findObject(comp), lab, opt)
			else:
				leg.AddEntry(comp, lab, opt)



	ylabel = plot_frame.GetYaxis().GetTitle()
	ylabel.replace("Events","Candidates")


	plot_frame.GetYaxis().SetTitle(ylabel)


	canvas.SaveAs(output_path + name + ".pdf[")
	plot_frame.Draw() # plot_frame is RooPlot
	if lhcbName: # lhcbName is TPaveText
		lhcbName.Draw("same")
	if leg: # leg is TLegend
		print("Drawing legend")
		leg.Print("v")
		leg.Draw("NCP")

	for obj,opt in additional_objects.items():
		obj.Print("v")
		print("Drawing additional object with option {}".format(opt))
		obj.Draw(opt)


	canvas.SaveAs(output_path + name + ".pdf")
	tmp_plot_minimum = plot_frame.GetMinimum()
	plot_frame.SetMinimum(y_min_log)
	canvas.SetLogy(1)
	plot_frame.Draw()
	if lhcbName:
		lhcbName.Draw("same")
	if leg:
		leg.Draw("NCP")

	for obj,opt in additional_objects.items():
		obj.Draw(opt)

	canvas.SaveAs(output_path + name + ".pdf")
	plot_frame.SetMinimum(tmp_plot_minimum)
	canvas.SetLogy(0)
	canvas.Clear()
	if not do_pulls:
		print("Pull plots are disabled.")
		canvas.SaveAs(output_path + name + ".pdf]")
		return

	#pull_hist_raw = plot_frame.pullHist()
	chi2_reduced = plot_frame.chiSquare(num_fit_params);
	ndof         = plot_frame.GetNbinsX()-num_fit_params;
	chi2_pvalue  = R.TMath.Prob(chi2_reduced*ndof, ndof);

	pull_hist_raw = _GetPulls(plot_frame)
	pull_hists = _color_coded_pull_hists(pull_hist_raw)
	_pull_distribution(pull_hist_raw, chi2_reduced, chi2_pvalue, output_path+name)
		#print("drawing {0} with option{1}".format(plottable_and_option[0], plottable_and_option[1]))
		#plottable_and_option[0].Print("v")
		#plottable_and_option[0].Draw(plottable_and_option[1])
	#R.gPad.RedrawAxis()
	#canvas.SaveAs(output_path + name + ".pdf")

	#tick_length = plot_frame.GetXaxis().GetTickLength()
	#plot_frame.GetXaxis().SetTickLength(0)

	pad_border = 0.02
	pad_relysplit = 0.35
	left_margin = 0.16

	top_label_size = 0.06
	top_title_offset = 1.2
	title2label_size_ratio = 1.1
	plot_frame.SetMinimum(y_min_log)

	pad_ysplit = (1.0-2.*pad_border)*pad_relysplit
	bottom_label_size = top_label_size*(1.-pad_relysplit)/pad_relysplit
	bottom_title_offset = top_title_offset/(1.-pad_relysplit)*pad_relysplit

	plot_frame.SetLabelSize(0.0,"x");
	plot_frame.SetLabelSize(top_label_size,"y");
	plot_frame.SetXTitle("");
	plot_frame.SetTitleSize(top_label_size*title2label_size_ratio,"y");
	plot_frame.GetYaxis().SetTitleOffset(top_title_offset);


	canvas.Divide(1,2)
	pad = canvas.cd(1)
	pad.SetPad(pad_border,pad_ysplit,1.-pad_border,1.-pad_border)
	pad.SetLeftMargin(left_margin)
	pad.SetBottomMargin(0.0)
	plot_frame.Draw("AH")
	plot_frame.GetYaxis().Draw("SAME")
	if lhcbName:
		lhcbName.Draw("same")
	if leg:
		leg.Draw("NCP")

	for obj,opt in additional_objects.items():
		obj.Draw(opt)

	pad = canvas.cd(2)
	pad.SetPad(pad_border,pad_border,1.-pad_border,pad_ysplit)
	pad.SetLeftMargin(left_margin)
	pad.SetTopMargin(0.0)
	pad.SetBottomMargin(0.4)


	plot_min = plot_frame.GetXaxis().GetXmin()
	plot_max = plot_frame.GetXaxis().GetXmax()
	pull_hist_raw.GetXaxis().SetLimits(plot_min,plot_max)
	pull_hist_raw.SetAxisRange(-5.8,5.8,"Y")
	pull_hist_raw.Draw()
	print("Setting X title to {0}".format(plot_variable.getTitle(True).Data() ))
	pull_hist_raw.GetXaxis().SetTitle(plot_variable.getTitle(True).Data() )
	
	pull_hist_raw.SetLabelSize(bottom_label_size, "xy");
	pull_hist_raw.SetTitleSize(bottom_label_size*title2label_size_ratio, "xy");
	pull_hist_raw.GetYaxis().SetTitleOffset(bottom_title_offset);  
	pull_hist_raw.GetYaxis().SetNdivisions(5,5,0);

	pull_hist_raw.GetXaxis().SetLabelOffset(0.005)
	#pull_hist_raw.SetTitle("")
	pull_hist_raw.SetYTitle("Pull")
	zero_line = R.TLine(plot_min, 0, plot_max, 0)

	lower_box = R.TBox(plot_min, -1., plot_max, -2.)
	upper_box = R.TBox(plot_min, +1., plot_max, +2.)

	upper_box.SetFillColor(11)
	upper_box.SetFillStyle(1001)
	upper_box.SetLineWidth(0)
	lower_box.SetFillColor(11)
	lower_box.SetFillStyle(1001)
	lower_box.SetLineWidth(0)

	pull_hist_raw.GetYaxis().SetRangeUser(-5.8,5.8)
	zero_line.Draw()
	upper_box.Draw()
	lower_box.Draw()

	#pull_hist_raw.GetXaxis().SetLabelOffset(999)
	for h in pull_hists:
		#h.SetLineWidth(2)
		h.SetAxisRange(-5.8,5.8,"Y")
		h.Draw("same")

	canvas.SaveAs(output_path + name + ".pdf")

	pad = canvas.cd(1)

	pad.SetLogy(1)
	plot_frame.Draw()
	if lhcbName:
		lhcbName.Draw("same")
	if leg:
		leg.Draw("NCP")

	for obj,opt in additional_objects.items():
		obj.Draw(opt)

	canvas.SaveAs(output_path + name + ".pdf")
	plot_frame.SetMinimum(tmp_plot_minimum)

	canvas.SaveAs(output_path + name + ".pdf]")
