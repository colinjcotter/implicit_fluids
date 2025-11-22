def get_initial_conditions(opts, u0, D0, b):
    testcase = opts.getString('testcase', 'w6')
    if testcase == 'w6':
        get_w6(u0, D0, b)
    else:
        raise NotImplementedError('testcase '+testcase)
