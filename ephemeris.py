"""Compute ephemeris of planetary position

Author: David H. Munro

See [JPL web page](https://ssd.jpl.nasa.gov/planets/approx_pos.html) for
an explanation of the formulas employed here.  The detailed source is
https://www.researchgate.net/publication/232203657 Standish and Williams
in book Explanatory Supplement to the Astronomical Almanac, Chapter 8
Publisher: University Science Books (2006).  The dwarf planet Pluto has
been reinstated here from that source.

Note that the coordinate frame is the solar system barycenter (CM), so
the Sun is not quite at the origin.  Planet masses from the detailed
source are included here to make this correction if desired.  Since
Jupiter is at about 5 AU and has about 1000th a solar mass, the CM is
displaced from the center of the Sun by roughly 0.005 AU, which is nearly
half the Sun's disk.  However, the inner planets experience almost the
same force from Jupiter as the Sun, so the centers of their orbits
tend to wobble along with the Sun, reducing errors in position of the
Sun due to this wobble.

See [Paul Schlyter's page](https://www.stjarnhimlen.se/comp/ppcomp.html)
for the model of the Moon implemented here.  Note that Schlyter uses a
time origin of 1999 Dec 31 00:00 UTC, which is day=-1.5 in the J2000 time
origin of 2000 Jan 1 12:00 UTC used in the JPL formulas.
Only the Sun and Moon models are used here.

API:

ephemeris - SolarSystem model to compute planetary orbits
JulianDate - class to convert Julian Dates to and from civil date and time
DateTime - subclass of python datetime supporting years <0001 and >9999
           (assuming proleptic Gregorian calendar extends forever)

=================================================
[MIT License](https://opensource.org/license/mit)

Copyright 2024 David H. Munro

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
“Software”), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from datetime import datetime, timezone
from numbers import Number

from numpy import array, asfarray, polyval, pi, sin, cos, sqrt

__all__ = ["ephemeris", "JulianDate", "DateTime", "sun2planet"]


class SolarSystem(object):
    """
    """
    def model(self, name, table, jdmin=None, jdmax=None):
        if jdmin is not None:
            jdmin = JulianDate(jdmin) + 0.  # + 0. converts to float
        if jdmax is not None:
            jdmax = JulianDate(jdmax) + 0.  # + 0. converts to float
        aux = None
        if not isinstance(table, str):
            table, aux = table
            aux_names, aux = self._parse_table(aux, False)
        table = table.replace("EM Bary", "Earth")  # fix name with a space
        names, table = self._parse_table(table)
        if "moon" in names:  # This is Schlyter table.
            # Adjust params to agree with JPL tables.
            # (ma, aper, nlon) --> (mlon, plon, nlon)
            table[4] += table[5]  # plon = aper + nlon
            table[10] += table[11]  # plondot = aperdot + nlondot
            table[3] += table[4]  # mlon = ma + plon
            table[9] += table[10]  # mlondot = madot + plondot
            # JPL time origin is 1.5 days later than Schlyter time origin.
            table[:6] += table[6:] * 1.5
            # JPL rates are per Julian century, Schlyter rates are per day.
            table[6:] *= 36525.0
            # Change Sun params to Earth params
        table[2:6] *= pi / 180.  # covert table degrees to radians
        table[8:] *= pi / 180.
        if aux is not None:
            aux *= pi / 180.  # covert aux degrees to radians

    @staticmethod
    def _parse_table(table, has_rates=True):
        lines = table.split("\n")
        lines = [line.split() for line in lines if line.strip()]
        if has_rates:
            lines = [a + b for a, b in zip(lines[0::2], lines[1::2])]
        names = [line[0].lower() for line in lines]
        values = [line[1:] for line in lines]
        values = array([[float(v) for v in vals] for vals in values]).T.copy()
        return names, values

    # The six orbital parameters in values and rates are:
    # a = ellipse semi-major axis
    # e = eccentricity
    # incl = orbital inclination
    # mlon = mean longitude
    # plon = longitude of perihelion
    # nlon = longitude of ascending node
    # values, rates shape = (nparams, nplanets)
    def orbit(self, t, values, rates, aux=None, auxlist=Ellipsis):
        # get values at times t, shape (nparams, nplanets, t.shape)
        values = values.reshape(values.shape + (1,)*t.ndim)
        values = values + rates.reshape(rates.shape + (1,)*t.ndim) * t
        a, e, incl, mlon, plon, nlon = values
        ma = mlon - plon  # mean anomaly
        if aux is not None:
            aux = aux.reshape(aux.shape + (1,)*t.ndim)
            ft = aux[3] * t
            # Some planets may not have aux, auxlist is indices into nplanets.
            ma[auxlist] += aux[0]*t**2 + aux[1]*cos(ft) + aux[2]*sin(ft)
        # compute eccentric anomaly ee by Newton iteration.
        ee, cee, see = self.kepler(e, ma)
        x, y = cee - e,  sqrt(1. - e**2) * see  # in orbital plane
        aper = plon - nlon  # argument of perihelion
        cw, sw = cos(aper), sin(aper)
        cn, sn = cos(nlon), sin(nlon)
        ci, si = cos(incl), sin(incl)
        cisn, cicn = ci*sn, ci*cn
        xyz = array([(cw*cn - sw*cisn)*x - (sw*cn + cw*cisn)*y,
                     (cw*sn + sw*cicn)*x - (sw*sn - cw*cicn)*y,
                     (sw*x + cw*y)*si])
        # Final result is a*xyz, but some code wants them separately.
        return xyz, a, e, ee, see, ma, aper, nlon

    @staticmethod
    def kepler(eps, ma, tol=1.e-9):
        """solve Kepler's equation ma = ee - eps*sin(ee) for ee given ma"""
        # This works as long as eps is less than and not too close to 1.
        # eps = eccentricity, ma = mean anomaly, ee = eccentric anonmaly
        # Tolerance tol is in radians of ma.
        ee = ma + eps*sin(ma + eps*sin(ma))  # initial guess
        while True:
            cee, see = cos(ee), sin(ee)
            dma = ma - (ee - eps*see)
            if (abs(dma) < tol).all():
                break  # Stop before changing ee so (cee, see) correct.
            ee += dma / (1. - eps*cee)
        return ee, cee, see


# ############ JulianDate and DateTime classes to handle time #############


# A simple alternative to the full-featured astropy.time.Time class.
class JulianDate(object):
    """Manage Julian Date including conversion to and from other formats.

    Examples::

        jd = JulianDate(julian_date_as_float)
        jd = JulianDate()  # jd for current time
        jd = JulianDate(instance_of_Julian_Date)
        jd = JulianDate("2024-04-29 03:26:02.11")  # ISO 8601 string
        jd = JulianDate(instance_of_datetime)
        jd = JulianDate("April 29, 2024  03:26",
                        "%B %d,  %Y %H:%M")  # parse using strptime()
        jd = JulianDate(posix=timestamp)  # POSIX time
        jd = JulianDate(j2000=djd)  # relative to J2000.0
        jd = JulianDate(astropy.time.Time instance)
             # useful for times beyond datetime calendar, like BCE

        jd.set(any of arguments accepted by constructor)
        jd += djd  # or jd -= djd
        # jd+djd or jd-djd in expressions are just floats

    Attributes
    ----------
    jd : float
        the Julian Date
    iso : string
        Julian Date in ISO 8601 format (UTC assumed)
    j2000 : float
        change in Julian Date since J2000.0
    """
    def __init__(self, date_time=None, format=None, **kwargs):
        """Initialize to specified date and time (see set_to method)."""
        self.set(date_time, format, **kwargs)

    def set(self, date_time=None, format=None, **kwargs):
        """Set to specified date and time.

        Parameters
        ----------
        date_time : Number or string or datetime, optional
            If a Number, covert to floating point JD.
            If a string, in an ISO 8601 format (e.g.- 2024-04-29 03:26:02.11)
            if no format argument is given.
            Can also be a python datetime object.  If the string does not
            specify a timezone, or if the datetime is naive, assumes UTC.
            Can also be an astropy.time.Time instance.
            If no date_time argument and there are no keyword arguments,
            set to the current UTC time.
            Otherwise, exactly one of the kwargs must be provided.
        format : string
            A format string passed to strptime() to parse date_time.  Unless
            an explicit time zone (%z directive) is included, the time is
            assumed to be UTC.
        j2000= : float, optional
            JD relative to J2000.0 (JD 2451545.0, 2000 Jan 1 12:00 UTC).
        posix= : float, optional
            POSIX timestamp (non-leap seconds since 1970 Jan 1 00:00 UTC).
        """
        j2000 = kwargs.get("j2000")
        if j2000 is not None:
            self.jd = 2451545.0 + float(j2000)
            return
        posix = kwargs.get("posix")
        if posix is not None:
            date_time = DateTime.fromtimestamp(float(posix), tz=timezone.utc)
        elif date_time is None:
            date_time = DateTime.now(timezone.utc)
        elif isinstance(date_time, Number):
            self.jd = float(date_time)
            return
        elif isinstance(date_time, JulianDate):
            self.jd = float(date_time.jd)
            return
        elif isinstance(date_time, str):
            if format is not None:
                date_time = DateTime.strptime(date_time, format)
            else:
                date_time = DateTime.fromisoformat(date_time)
            date_time = date_time.astimezone(timezone.utc)
        elif hasattr(date_time, "jd1") and hasattr(date_time, "to_value"):
            # Handle astropy.time.Time instance.
            try:
                self.jd = date_time.to_value("jd", "float")
                return
            except Exception:
                pass  # Gave it a try, give up.
        # Assume date_time is a datetime instance (possibly a DateTime).
        if (date_time.tzinfo is None
                or date_time.tzinfo.utcoffset(date_time) is None):
            # Assume naive datetime is UTC.
            date_time = date_time.replace(tzinfo=timezone.utc)
        # 86400 sec/day, JD at 1970-1-1 00:00 UTC is POSIX timestamp
        self.jd = date_time.timestamp()/86400. + 2440587.5

    @property
    def datetime(self):
        """time as python datetime object (UTC timezone)"""
        return DateTime.fromtimestamp((self.jd - 2440587.5)*86400.,
                                      timezone.utc)

    @property
    def iso(self):
        """UTC time in ISO 8601 format"""
        return self.datetime.replace(tzinfo=None).isoformat(" ")

    @property
    def j2000(self):
        """Julian Date relative to J2000.0"""
        return self.jd - 2451545.0

    def strftime(self, format):
        """return time formatted as strftime(format)"""
        return self.datetime.strftime(format)

    def __format__(self, format):
        return self.datetime.__format__(format)

    def __repr__(self):
        return "JulianDate(" + repr(float(self.jd)) + ")"

    def __str__(self):
        return str(self.jd)

    def __float__(self):
        return float(self.jd)

    # Permit +, -, +=, -= operations on JulianDate instances
    def __add__(self, djd):
        return self.jd + djd

    def __sub__(self, djd):
        if isinstance(djd, JulianDate):  # can subtract two JulianDate-s
            djd = djd.jd
        return self.jd - djd

    def __radd__(self, djd):
        return self.jd + djd

    def __rsub__(self, jd):
        return jd - self.jd

    def __iadd__(self, djd):
        self.jd += djd

    def __isub__(self, djd):
        self.jd -= djd

    # FIXME: should also have comparison operators here


class DateTime(datetime):
    """Simple extension to datetime with extended MINYEAR and MAXYEAR support.

    Python datetime very sensibly only handles years from 0001 to 9999.
    There are exactly 146097 days every 400 years in the proleptic Gregorian
    calendar, so an easy way to handle dates outside this range is to shift
    a given date by a multiple of 146097 days when it falls outside the
    natively supported range.  Note that since 146097 is divisible by 7,
    a 400 year shift keeps the days of the week intact.

    Only fromisoformat, isoformat, fromtimestamp, timestamp, replace,
    astimezone methods, the DateTime constructor, and the year
    attribute can handle the extended year range.

    Also DateTime adds a jd attribute and a fromjd(jd) class method.

    """
    __slots__ = ("shift400",)

    # Since datetime instance is immutable, needs __new__, not __init__.
    def __new__(cls, year, *args, **kwargs):
        year, shift400 = cls._shift_year(year)
        # Unlike other class methods, __new__ requires explicit cls arg??
        dt = super(DateTime, cls).__new__(cls, year, *args, **kwargs)
        dt.shift400 = shift400
        return dt

    @staticmethod
    def _shift_year(year):
        shift400 = 0
        if year < 2:  # not 1 to avoid astimezone conversion errors
            shift400 = year // 400 if year else -1
        elif year > 9998:  # not 9999 to avoid astimezone conversion errors
            shift400 = year // 400 - 23
        return year - 400*shift400, shift400

    @property
    def year(self):
        y = super(DateTime, self).year
        if self.shift400:
            y += 400 * self.shift400
        return y

    def isoformat(self, *args, **kwargs):
        iso = super(DateTime, self).isoformat(*args, **kwargs)
        if self.shift400:
            year = int(iso[:4]) + 400 * self.shift400
            iso = ("-" if year < 0 else "") + "{:04d}" + iso[4:]
            iso = iso.format(abs(year))
        return iso

    @classmethod
    def fromisoformat(cls, date_string):
        minus = date_string.startswith("-")
        if minus:
            date_string = date_string[1:]
        year, date_string = date_string.split("-", 1)
        year = int(year)
        year, shift400 = cls._shift_year(-year if minus else year)
        date_string = "{:04d}-".format(year) + date_string
        dt = super(DateTime, cls).fromisoformat(date_string)
        kwargs = dict(fold=dt.fold) if hasattr(dt, "fold") else {}
        jd = DateTime(year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
                      dt.microsecond, dt.tzinfo, **kwargs)
        jd.shift400 = shift400
        return jd

    def timestamp(self, *args, **kwargs):
        posix = super(DateTime, self).timestamp(*args, **kwargs)
        if self.shift400:
            posix += 86400. * 146097. * self.shift400
        return posix

    @classmethod
    def fromtimestamp(cls, timestamp, tz=None):
        jd = timestamp/86400.0 + 2440587.5
        if jd < 1721790.5:  # 0002-01-01 00:00:00
            shift400 = int((jd - 1721790.5) // 146097)  # days in 400 years
            if shift400 == 0:
                shift400 = -1
        elif jd >= 5373119.5:  # 9999-01-01 00:00:00
            shift400 = int((jd - 5373119.5) // 146097) + 1  # days in 400 years
        else:
            shift400 = 0
        jd -= 146097.0 * shift400
        timestamp = (jd - 2440587.5) * 86400.0
        dt = datetime.fromtimestamp(timestamp, tz)
        if hasattr(dt, "fold"):
            kwargs = dict(fold=dt.fold)
        else:
            kwargs = {}
        dt = DateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                      dt.second, dt.microsecond, dt.tzinfo, **kwargs)
        dt.shift400 = shift400
        return dt

    def replace(self, *args, **kwargs):
        if args:
            year, args = args[0], args[1:]
        else:
            year = kwargs.pop("year", self.year)
        year, shift400 = self._shift_year(year)
        dt = super(DateTime, self).replace(year, *args, **kwargs)
        dt.shift400 = self.shift400
        return dt

    def astimezone(self, *args, **kwargs):
        dt = super(DateTime, self).astimezone(*args, **kwargs)
        dt.shift400 = self.shift400
        return dt

    def __repr__(self):  # __str__ uses isoformat __repr__ does not
        return "DateTime.fromisoformat('" + str(self) + "')"

    @property
    def jd(self):
        return self.timestamp()/86400.0 + 2440587.5

    @classmethod
    def fromjd(cls, jd):
        return cls.fromtimestamp((jd - 2440587.5) * 86400.0, timezone.utc)


# ############ Load data for JPL and Schlyter solar system models #############


ephemeris = SolarSystem()

# JPL models
# columns: planet, a, e, incl, mlon, plon, nlon
#          rates are per Julian-century (36525 days)
ephemeris.model("JPL short", """
Mercury   0.38709927      0.20563593      7.00497902      252.25032350     77.45779628     48.33076593
          0.00000037      0.00001906     -0.00594749   149472.67411175      0.16047689     -0.12534081
Venus     0.72333566      0.00677672      3.39467605      181.97909950    131.60246718     76.67984255
          0.00000390     -0.00004107     -0.00078890    58517.81538729      0.00268329     -0.27769418
EM Bary   1.00000261      0.01671123     -0.00001531      100.46457166    102.93768193      0.0
          0.00000562     -0.00004392     -0.01294668    35999.37244981      0.32327364      0.0
Mars      1.52371034      0.09339410      1.84969142       -4.55343205    -23.94362959     49.55953891
          0.00001847      0.00007882     -0.00813131    19140.30268499      0.44441088     -0.29257343
Jupiter   5.20288700      0.04838624      1.30439695       34.39644051     14.72847983    100.47390909
         -0.00011607     -0.00013253     -0.00183714     3034.74612775      0.21252668      0.20469106
Saturn    9.53667594      0.05386179      2.48599187       49.95424423     92.59887831    113.66242448
         -0.00125060     -0.00050991      0.00193609     1222.49362201     -0.41897216     -0.28867794
Uranus   19.18916464      0.04725744      0.77263783      313.23810451    170.95427630     74.01692503
         -0.00196176     -0.00004397     -0.00242939      428.48202785      0.40805281      0.04240589
Neptune  30.06992276      0.00859048      1.77004347      -55.12002969     44.96476227    131.78422574
          0.00026291      0.00005105      0.00035372      218.45945325     -0.32241464     -0.00508664
Pluto    39.48211675      0.24882730     17.14001206      238.92903833    224.06891629    110.30393684
         -0.00031596      0.00005170      0.00004818      145.20780515     -0.04062942     -0.01183482
""", "1800-01-01 12:00:00", "2050-01-01 12:00:00")

ephemeris.model("JPL long",
                ("""
Mercury   0.38709843      0.20563661      7.00559432      252.25166724     77.45771895     48.33961819
          0.00000000      0.00002123     -0.00590158   149472.67486623      0.15940013     -0.12214182
Venus     0.72332102      0.00676399      3.39777545      181.97970850    131.76755713     76.67261496
         -0.00000026     -0.00005107      0.00043494    58517.81560260      0.05679648     -0.27274174
EM Bary   1.00000018      0.01673163     -0.00054346      100.46691572    102.93005885     -5.11260389
         -0.00000003     -0.00003661     -0.01337178    35999.37306329      0.31795260     -0.24123856
Mars      1.52371243      0.09336511      1.85181869       -4.56813164    -23.91744784     49.71320984
          0.00000097      0.00009149     -0.00724757    19140.29934243      0.45223625     -0.26852431
Jupiter   5.20248019      0.04853590      1.29861416       34.33479152     14.27495244    100.29282654
         -0.00002864      0.00018026     -0.00322699     3034.90371757      0.18199196      0.13024619
Saturn    9.54149883      0.05550825      2.49424102       50.07571329     92.86136063    113.63998702
         -0.00003065     -0.00032044      0.00451969     1222.11494724      0.54179478     -0.25015002
Uranus   19.18797948      0.04685740      0.77298127      314.20276625    172.43404441     73.96250215
         -0.00020455     -0.00001550     -0.00180155      428.49512595      0.09266985      0.05739699
Neptune  30.06952752      0.00895439      1.77005520      304.22289287     46.68158724    131.78635853
          0.00006447      0.00000818      0.00022400      218.46515314      0.01009938     -0.00606302
Pluto    39.48686035      0.24885238     17.14104260      238.96535011    224.09702598    110.30167986
          0.00449751      0.00006016      0.00000501      145.18042903     -0.00968827     -0.00809981
""",
                 # auxilliary table for longer time spans
                 # columns: planet, b, c, s, f
                 """
Jupiter   -0.00012452    0.06064060   -0.35635438   38.35125000
Saturn     0.00025899   -0.13434469    0.87320147   38.35125000
Uranus     0.00058331   -0.97731848    0.17689245    7.67025000
Neptune   -0.00041348    0.68346318   -0.10162547    7.67025000
Pluto     -0.01262724    0.0           0.0           0.0
"""))

# Schlyter models
# columns: planet, a, e, incl, ma, aper, nlon
#          rates are per day
ephemeris.model("Schlyter", """
Sun      1.000000   0.016709  0.0000    356.0470      282.9404      0.00000
         0.000000  -1.151e-9  0.0000    0.9856002585  4.70935E-5    0.00000
Moon     60.2666    0.054900  5.1454    115.3654      318.0634      125.1228
         0.00000    0.000000  0.0000    13.0649929509 0.1643573223 -0.0529538083
Mercury  0.387098   0.205635  7.0047    168.6562      29.1241       48.3313
         0.000000   5.59E-10  5.00E-8   4.0923344368  1.01444E-5    3.24587E-5
Venus    0.723330   0.006773  3.3946    48.0052       54.8910       76.6799
         0.000000  -1.302E-9  2.75E-8   1.6021302244  1.38374E-5    2.46590E-5
Mars     1.523688   0.093405  1.8497    18.6021       286.5016      49.5574
         0.000000   2.516E-9 -1.78E-8   0.5240207766  2.92961E-5    2.11081E-5
Jupiter  5.20256    0.048498  1.3030    19.8950       273.8777      100.4542
         0.000000   4.469E-9 -1.557E-7  0.0830853001  1.64505E-5    2.76854E-5
Saturn   9.55475    0.055546  2.4886    316.9670      339.3939      113.6634
         0.000000  -9.499E-9 -1.081E-7  0.0334442282  2.97661E-5    2.38980E-5
Uranus   19.18171   0.047318  0.7733    142.5905      96.6612       74.0005
        -1.55E-8    7.45E-9   1.9E-8    0.011725806   3.0565E-5     1.3978E-5
Neptune  30.05826   0.008606  1.7700    260.2471      272.8461      131.7806
         3.313E-8   2.15E-9  -2.55E-7   0.005995147  -6.027E-6      3.0173E-5
""", )

# reciprocal planet masses, mass_sun / mass_planet
sun2planet = dict(
    mercury=6023600., venus=408523.71, earth=328900.5614, mars=3098708.,
    jupiter=1047.3486, saturn=3497.898, uranus=22902.98, neptune=19412.24,
    pluto=135200000.)


def jd2t(jd):
    """convert Julian date to T in centuries past J2000.0"""
    return (asfarray(jd) - 2451545.) / 36525.


# estimated accuracy is <0.02 as for 1000 yr, a few as after 10000 yr
# Obliquity at J2000.0 (23 26 21.448) = 23.439291 deg = 84381.448 as
def obliquity(jd):
    """mean obliquity of the ecliptic (Laskar) in degrees"""
    t = 0.01 * jd2t(jd).clip(-100., 100.)  # only valid for +-10000 yr
    return polyval(_LASKAR, t)


# leading term is 71.583433 yr/deg
# Obliquity at J2000.0 (23 26 21.448) = 23.439291 deg = 84381.448 as
def precession(jd):
    """precession of the equinoxes (Laskar) in degrees"""
    t = 0.01 * jd2t(jd).clip(-100., 100.)  # only valid for +-10000 yr
    return polyval(_LASKARP, t)


_LASKAR = array([2.45, 5.79, 27.87, 7.12, -39.05, -249.67, -51.38, 1999.25,
                 -1.55, -4680.93, 84381.448]) / 3600.  # deg/100 J-century

_LASKARP = array([-8.66, -47.59, 24.24, 130.95, 174.51, -180.55, -2352.16,
                  77.32, 11119.71, 502909.66]) / 3600.  # deg/100 J-century
