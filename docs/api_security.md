# API Security
Strict endpoint inventory. All state-mutating requests require Bearer tokens. Invalid tokens yield 401. Unauthorized cross-session access yields 403.