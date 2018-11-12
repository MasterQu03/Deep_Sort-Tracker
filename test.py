def lengthOfLongestSubstring(s):
    d_map = {}
    start = maxLength = 0
    for i in range(len(s)):
        if s[i] in d_map and start <= d_map[s[i]]:
            start = d_map[s[i]] + 1
        else:
            maxLength = max(maxLength, i - start + 1)
        d_map[s[i]] = i
    return maxLength


s = "AGCTAGCT"
print(lengthOfLongestSubstring(s))