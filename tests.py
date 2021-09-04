# Python code for calculating number of ways
# to distribute m mangoes amongst n people
# where all mangoes and people are identical


# function used to generate binomial coefficient
# time complexity O(m)
def binomial_coefficient(n, m):
    res = 1

    if m > n - m:
        m = n - m

    for i in range(0, m):
        res *= (n - i)
        res /= (i + 1)

    return res


# helper function for generating no of ways
# to distribute m mangoes amongst n people
def calculate_ways(m, n):
    # not enough mangoes to be distributed
    if m < n:
        return 0

    # ways  -> (n + m-1)C(n-1)
    ways = binomial_coefficient(n + m - 1, n - 1)
    return int(ways)


# Driver function
if __name__ == '__main__':
    # m represents number of mangoes
    # n represents number of people
    m = 16
    n = 4

    result = calculate_ways(m, n)
    print(result)