function [e] = testPassVar(e)
    e
    sum(sum(e{1,2}{1,1}))
    sum(sum(e{1,2}{1,2}))
return 