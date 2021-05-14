import argparse

from astropy import units
from typing import Tuple

from sigpyproc.header import Header


def get_duration_print(duration: float) -> Tuple[str, float]:
    print_unit = "seconds"
    if duration > 60:
        duration /= 60
        print_unit = "minutes"
        if duration > 60:
            duration /= 60
            print_unit = "hours"
            if duration > 24:
                duration /= 24
                print_unit = "days"
    return print_unit, duration


def header_string(header: Header) -> str:
    print_hdr = []
    temp = "{0:<33}: {1}"
    print_hdr.extend(
        [
            temp.format("Data file", header.rawdatafile),
            temp.format("Header size (bytes)", header.hdrlens[0]),
            temp.format("Data size (bytes)", header.datalens[0]),
            temp.format("Data type", f"{header.data_type} ({header.frame})"),
            temp.format("Telescope", header.telescope),
            temp.format("Datataking Machine", header.backend),
            temp.format("Source Name", header.source),
            temp.format("Source RA (J2000)", header.ra),
            temp.format("Source DEC (J2000)", header.dec),
            temp.format("Start AZ (deg)", header.azimuth),
            temp.format("Start ZA (deg)", header.zenith),
        ]
    )

    if header.data_type == "filterbank":
        print_hdr.extend(
            [
                temp.format("Frequency of channel 1 (MHz)", header.fch1),
                temp.format("Channel bandwidth      (MHz)", header.foff),
                temp.format("Number of channels", header.nchans),
                temp.format("Number of beams", header.nbeams),
                temp.format("Beam number", header.ibeam),
            ]
        )
    elif header.data_type == "time series":
        print_hdr.extend(
            [
                temp.format("Reference DM (pc/cc)", header.dm),
                temp.format("Reference frequency    (MHz)", header.fch1),
                temp.format("Number of channels", header.nchans),
            ]
        )

    print_unit, print_dur = get_duration_print(header.tobs)
    print_hdr.extend(
        [
            temp.format("Time stamp of first sample (MJD)", header.tstart),
            temp.format("Gregorian date (YYYY-MM-DD)", header.obs_date),
            temp.format(
                "Sample time (us)",
                (header.tsamp * units.second).to(units.microsecond).value,
            ),
            temp.format("Number of samples", header.nsamples),
            temp.format(f"Observation length {print_unit}", f"{print_dur:.1f}"),
            temp.format("Number of bits per sample", header.nbits),
            temp.format("Number of IFs", header.nifs),
        ]
    )
    return "\n".join(print_hdr)


def main():
    description = "Examine header parameters of filterbank data."
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("filename", type=str, help="Path of the filterbank data file")

    parser.add_argument(
        "-k",
        "--key",
        type=str,
        help="A header key to read (e.g. telescope, fch1, nsamples)",
    )

    args = parser.parse_args()

    header = Header.from_sigproc(args.filename)
    if args.key:
        print(f"{args.key} = {getattr(header, args.key)}", flush=True)  # noqa: WPS421
    else:
        print(header_string(header), flush=True)  # noqa: WPS421


if __name__ == "__main__":
    main()
