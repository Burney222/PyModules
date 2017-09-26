'''
Module containing small general helper functions
'''

def ensure_dir(directory):
	"""When directory is not present, create it.

	Arguments: 
	directory: name of directory.
	"""
	import os
	if not os.path.exists(directory):
		os.makedirs(directory)

#=====================+===============
#======================+==============
def printBumper(text, c='=', n=-1):
	"""Print a text with separating character. 

	Arguments: 
	text: text to be printed between separating character.

	c: separating character. 

	n: number of printed characters prior to the text (default -1: print as many characters as length of text)."""
	if n==-1:
		times = len(text)
	else:
		times = n
	print(c*times)
	print(text, ( (times-len(text)-1)*c ))
	print(c*times)
	