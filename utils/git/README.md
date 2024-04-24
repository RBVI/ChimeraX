This directory contains _optional_ files that can be used by developers contributing to
ChimeraX to make the git workflow a little easier.

- gitmessage

  Inserts the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) message template into new commits

- commit-msg

  Verifies that the commit message follows the Conventional Commit spec by running `commitlint`.
  Install `commitlint` with `npm install -g @commitlint/{config-conventional,cli}`.

- commitlintconfig.js

  The configuration file for commitlint.

- pre-commit
  Looks at changes in bundles and verifies that those bundles have their version numbers bumped.
