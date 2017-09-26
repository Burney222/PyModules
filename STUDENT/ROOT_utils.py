from utilities import ensure_dir

def saveCanvas(can, name):
	"""Saves Canvas in plots/ folder as pdf/*.pdf (and C/root/png).

	Arguments:
	can: TCanvas to be saved.

	name: Name of the files: e.g. name='Trigger' --> plots/pdf/Trigger.pdf'.
	"""
	ensure_dir("plots/C")
	ensure_dir("plots/pdf")
	ensure_dir("plots/png")
	ensure_dir("plots/root")
	can.SaveAs("plots/C/"+name+".C")
	can.SaveAs("plots/png/"+name+".png")
	can.SaveAs("plots/pdf/"+name+".pdf")
	can.SaveAs("plots/root/"+name+".root")
	