===================
Working with Github
===================


Setting it up right
-------------------

We assume that any contributor has a Github account and established a secure
connection using the GITHUB_TOKEN or SSH. If you wish to set up a new
account or want to know how to set up a secure connection before contributing,
please see here:

`Getting started with Github <https://docs.github.com/en/get-started>`_,
and `creating a personal access token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`_.

Fork and clone the Espresso repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _fork_clone:

1. Navigate to the `GitHub repository <https://github.com/inlab-geo/espresso>`_
2. Click the "Fork" button on top right of the page (followed by a confirmation page
   with a "Create fork" button)
3. Now you will be redirected to your own fork of Espresso,
   where you can freely commit changes and add your code.

   .. mermaid::

     %%{init: { 'logLevel': 'debug', 'theme': 'base', 'gitGraph': {'showCommitLabel': false}} }%%
       gitGraph
         commit
         commit
         branch your_own_fork
         checkout your_own_fork
         commit
         commit
         checkout main
         merge your_own_fork
         commit
         commit

4. Next, create a local copy of your fork of the Espresso repository using "clone"
   (the Github equivalent of download)::

     git clone https://github.com/YOUR_GITHUB_ACCOUNT/espresso.git

   replacing YOUR_GITHUB_ACCOUNT with your own account.

