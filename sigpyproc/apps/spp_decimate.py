import argparse

from sigpyproc.readers import FilReader


def decimate(filename, tfactor=1, ffactor=1, gulp=16384, outfile=None):
    fil = FilReader(filename)
    fil.downsample(tfactor=tfactor, ffactor=ffactor, gulp=gulp, outfile=outfile)


def main():
    description = "Reduce time and/or frequency resolution of filterbank data."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("filename", type=str, help="Path of the filterbank data file")

    parser.add_argument("-o", "--outfile", type=str, help="Output file name")

    parser.add_argument(
        "-f", "--nchan", type=str, default=1, help="Number of frequency channels to add"
    )

    parser.add_argument(
        "-t", "--nsamp", type=str, default=1, help="Number of time samples to add"
    )

    parser.add_argument("--nbits", type=str, help="Output number of bits per sample")

    parser.add_argument(
        "-g", "--gulp", type=str, default=16384, help="Number of samples to read at once"
    )
    args = parser.parse_args()

    decimate(
        args.filename,
        tfactor=args.samp,
        ffactor=args.nchan,
        gulp=args.gulp,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
