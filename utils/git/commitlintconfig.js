module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    // name - configuration array pairs: [strictness, applicability, value]
    // strictness: 0 disables, 1 issues warning on violation, 2 errors out on violation
    // applicability: always to apply the rule, never to apply and invert the rule e.g. "never over 80 columns"
    // value: the value to apply to the rule, e.g. 80 for the column rule
    "scope-case": [1, "always", "lower-case"],
    "subject-case": [1, "always", ["sentence-case"]],
    "type-enum": [
      2,
      "always",
      [
        "feat",
        "fix",
        "revert",
        "build",
        "chore",
        "ci",
        "docs",
        "style",
        "test",
        "dev",
      ],
    ],
  },
};
