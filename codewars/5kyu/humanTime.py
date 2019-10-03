from codewars.test import Test

def make_readable(seconds):

    secs = seconds % 60
    min = (seconds // 60)
    mins =  min % 60
    hr = (min // 60)
    hrs = hr

    shr = "{:02d}".format(hrs)
    smin = "{:02d}".format(mins)
    ssecs = "{:02d}".format(secs)

    time = shr + ":" + smin + ":" + ssecs
    print(time)
    return time





Test = Test()

Test.assert_equals(make_readable(0), "00:00:00")
Test.assert_equals(make_readable(5), "00:00:05")
Test.assert_equals(make_readable(60), "00:01:00")
Test.assert_equals(make_readable(86399), "23:59:59")
Test.assert_equals(make_readable(359999), "99:59:59")