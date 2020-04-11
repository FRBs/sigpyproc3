from sys import argv
from sigpyproc.Readers import FilReader
fil = FilReader(argv[1])
tim = fil.dedisperse(float(argv[2])).toFile(f"{fil.header.basename}_DM{argv[2]}.tim")
