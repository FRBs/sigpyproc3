from sys import argv
from sigpyproc.Readers import FilReader
fil = FilReader(argv[1])
tim = fil.dedisperse(float(argv[2])).toFile("%s_DM%s.tim"%(fil.header.basename,argv[2]))
