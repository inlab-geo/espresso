================
Submit on GitHub
================

Commit & push
-------------

#. It's helpful to "commit" your changes when you have any progress. Feel free to make 
   commits as often as necesary. Check 
   `this cheatsheet <https://education.github.com/git-cheat-sheet-education.pdf>`_
   for a good reference of using Git.

   To commit a new contribution to the repository, we therefore recommend to use
   the following commands:

   .. code-block:: console

        $ git add contrib/<new-problem> # Adds the new folder, but no other changes
        $ git commit -m "feat: My commit message"

   Please note that we aim to use
   `Angular style <https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#-git-commit-guidelines>`_
   commit messages throughout our projects. Simply speaking, we categorise our commits by
   a short prefix (from ``feat``, ``fix``, ``docs``, ``style``, ``refactor``, ``perf``,
   ``test`` and ``chore``).

#. Once your changes are committed, push the commits into your remote fork:

   .. code-block:: console

        $ git push origin main

#. In your remote repository under your GitHub account you should be able to see
   your new commits.


Raise a pull request
--------------------

#. Now that you've finished the coding and editing work, open your 
   fork on GitHub :code:`https://github.com/<your-gh-account>/espresso` in a browser.

#. Find the word "Contribute" on top of the page, click it and choose the green "Open 
   pull request" button. Follow the prompts and fill in necessary message you'd like us
   to know.

   .. figure:: ../_static/contrib_pr1.png
    :align: center

#. Once your pull request is submitted, some automatic checks will be triggered. Rest 
   assured - we will review your contribution, comment if necessary, and proceed to merge
   your contribution into our main repository when everything's ready.

#. After your contribution is merged to the main branch, you can request another change
   with the same workflow anytime you want. Just keep your own fork, edit, commit and 
   push to your own fork, and raise a pull request from there.

#. Thanks again for your contribution 🌟 
