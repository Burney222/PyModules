# SciFi related functions


def coursefine2time(course, fine):
    """Function to convert course and fine value to time (in ps)"""
    return 781.25*course + 48.8*fine


def time2coursefine(time):
    """Function to convert time (in ps) to (course, fine) value pair"""
    course = time//781.25
    fine = (time%781.25)/48.8
    return (course, fine)
