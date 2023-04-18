=============
Set up GitHub
=============

General workflow
----------------

.. mermaid::

  %%{init: {'theme':'base'}}%%
    flowchart LR
      subgraph PREPARATION [ preparation ]
        direction TB
        fork(fork repository)-->clone(create local clone)
        clone-->env_setup(environment setup)
      end
      subgraph EDIT [ editing ]
        direction TB
        code(start coding)-->commit(commit as needed)
        commit-->push(push to your own fork)
      end
      subgraph SUBMIT [ submission ]
        direction TB
        pr(create pull request)-->modify(edit based on our comments)
        modify-->commit_push(commit and push)
        commit_push-->merge(we merge it once ready)
        pr-->merge
      end
      PREPARATION-->EDIT
      EDIT-->SUBMIT


Setting up GitHub
-----------------

We assume that any contributor has a Github account and established a secure
connection using the GITHUB_TOKEN or SSH. If you wish to set up a new
account or want to know how to set up a secure connection before contributing,
please see here: `Getting started with Github <https://docs.github.com/en/get-started>`_.

.. dropdown:: Instructions for first-time GitHub users
    :icon: info

    If this is the first time you clone a GitHub repository, it's very likely that you 
    will need a personal access token as your password. 
    
    **Option 1** - Check out this page:
    `creating a personal access token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`_
    for how to create a personal access token, and use it as your password when you are
    asked to enter it in the terminal.

    **Option 2** - Alternatively, set up SSH key and upload your public key to your 
    GitHub account. Follow instructions in this page:
    `Generating a new SSH key and adding it to the ssh-agent <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`_
    for how to set up SSH keys with GitHub.


Fork and clone
--------------

#. Navigate to the `GitHub repository <https://github.com/inlab-geo/espresso>`_
#. Click the "Fork" button on top right of the page (followed by a confirmation page
   with a "Create fork" button)

   .. figure:: ../_static/contrib_fork.png
    :align: center

   .. figure:: ../_static/contrib_fork2.png
    :align: center

#. Now you will be redirected to your own fork of Espresso,
   where you can freely commit changes and add your code.

   .. figure:: ../_static/contrib_fork3.png
    :align: center

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

#. Next, create a local copy of your fork of the Espresso repository using "clone"
   (the Github equivalent of download):

   .. figure:: ../_static/contrib_fork4.png
    :align: center

   .. code-block:: console

     $ git clone https://github.com/YOUR_GITHUB_ACCOUNT/espresso.git
     $ git remote add upstream https://github.com/inlab-geo/espresso.git
     $ git fetch upstream

   replacing YOUR_GITHUB_ACCOUNT with your own account.

#. Open your local copy (folder ``espresso``) using your favourite editor (VSCode,
   Spyder, etc.). Continue with :doc:`setting up development environment <setup>`.
