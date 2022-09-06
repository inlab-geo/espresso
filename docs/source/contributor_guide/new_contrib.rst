====================
New Espresso problem
====================

👋 Everyone is welcomed to contribute with their own forward code. We aim to reduce the
barrier of contributing so don't worry if you are not familiar with those technical
stuff like git operations - we are here to help.

There are generally three steps involved in submiting your code:

- **Download** - :ref:`get a copy of Espresso <get_own_copy>`
- **Edit** - :ref:`add in your own Espresso problem <add_contrib>`
- **Upload** - :ref:`submit your changes to our main repository <submit_changes>`

In the following paragraphs, we are going to show you how to complete each of the steps
above. Again, feel free to `contact us <../user_guide/faq.html>`_ when in doubt.


.. _get_own_copy:

Get your own copy of Espresso
-----------------------------

#. Open your browser and go to the Espresso `repository`_.
#. Ensure you have a GitHub account and it's signed in. If not, click the "Sign Up"
   button on the top right and fill in the necessary information to sign up an account.
#. Now click the "Fork" button on the top right.

   .. figure:: ../_static/contrib_fork.png
    :align: center

#. Leave everything by default and click the green "Create fork" button.

   .. figure:: ../_static/contrib_fork2.png
    :align: center

#. Now you will be redirected to your own "fork" of the Espresso repository.

   This fork is your own version of Espresso, and you can make changes however you 
   want. We will later demonstrate that after you make your own changes, you are
   able to "contribute" back to the main repository.

   .. figure:: ../_static/contrib_fork3.png
    :align: center

#. We will clone your fork into your local machine. Click the green "Code" button first, 
   and then copy the content under the "HTTPS" tab.

   .. figure:: ../_static/contrib_fork4.png
    :align: center

#. Clone your fork to somewhere in your computer.

   - For **MacOS** and **Linux** users, open your "Terminal" app, change your working 
     directory into somehwere you'd like to place the Espresso code, then run the 
     :code:`git clone` command as following.
   - For **Windows** users, please install `git <https://git-scm.com/downloads>`_ first, 
     and open "Git Bash" to run the following commands. In the steps afterwards, it's
     always "Git Bash" when we refer to a "terminal" if you are on Windows.

   .. code-block:: console

    cd <path-to-espresso>
    git clone <url-you-copied-in-step-6>

   .. admonition:: Instructions for first-time GitHub users
      :class: dropdown, attention

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

#. Open :code:`<path-to-espresso>/espresso` folder with your favourite code editor. 
   You will see a copy of Espresso in front of you, cheers ☕️! 


.. _add_contrib:

Add your own Espresso problem
-----------------------------

#. Let's now ensure that you have a correct environment set up. Python >= 3.6 is required,
   and see this 
   `environment_contrib.yml <https://github.com/inlab-geo/espresso/blob/main/envs/environment_contrib.yml>`_ 
   file for a list of required packages.

   .. toggle::
        
        - Choose a Python environment manager first. 
          `mamba <https://mamba.readthedocs.io/en/latest/>`_ /
          `conda <https://docs.conda.io/en/latest/>`_ is recommended as it can set 
          up system-wide dependencies as well, but feel free to use the one you are most 
          familiar with.

        - Python >= 3.6 is required.

        - If you use `mamba <https://mamba.readthedocs.io/en/latest/>`_ /
          `conda <https://docs.conda.io/en/latest/>`_, run 
          :code:`conda create -f envs/environment_contrib.yml` under the project root folder.
          Otherwise, make sure you have the list of packages in 
          `environment_contrib.yml <https://github.com/inlab-geo/espresso/blob/main/envs/environment_contrib.yml>`_
          in the virtual environment with your preferred tool.

#. Install Espresso core library - this enables you to access the base class for an Espresso problem
   :code:`EspressoProblem` and some utility functions to help the development.

   Run the following in your terminal, with :code:`<path-to-espresso>/` as your working directory.

   .. code-block:: bash

      pip install .

#. Create a folder for your new contribution under :code:`contrib/<problem-name>`,
   by running the following in your terminal:

   .. code-block:: bash

        python <path-to-espresso>/tools/new_contribution/create_new_contrib.py <problem-name>

   Replacing :code:`path-to-espresso` with your path to the espresso folder you've just cloned,
   and :code:`problem-name` with your Espresso problem name, with lower case words connected
   by underscores (e.g. :code:`gravity_density`, :code:`polynomial_regression`).

#. Navigate to folder :code:`<path-to-espresso>/contrib/<problem-name>`, and you'll see template 
   files.

   .. figure:: ../_static/contrib_edit1.png
    :align: center

#. Read instructions in the :code:`README.md` file, and you will know what to do next 🧑🏽‍💻👩🏻‍💻👨‍💻

   #. You should already have all the "pre-requisites" installed if you've gone through 
      the steps above.

   #. Check the boxes under "getting started". These are pretty much all the things you've
      got to do to complete this contribution.

   #. When you'd like to perform a quick local test by running your own code, tips under
      "how to unit test your code" can be useful.

   #. When you think you've finished the coding, use scripts under "how to test building your
      contribution with :code:`cofi-expresso`" to include your contribution into the package
      locally.


.. _submit_changes:

Submit your changes
-------------------

#. It's helpful to "commit" your changes when you have any progress. Feel free to make 
   commits as often as necesary.
   
   - Use :code:`git add <file-name-1> <file-name-2>` to choose which files you'd like to 
     include in the following "commit".
   - Use :code:`git commit -m "progress in xxx"` to commit your changes.
   - Use :code:`git push origin <branch-name>` to push your changes onto your GitHub fork,
     where :code:`<branch-name>` is :code:`main` by default.

   .. seealso::

    Check `this cheatsheet <https://education.github.com/git-cheat-sheet-education.pdf>`_
    for a good reference of using Git.

#. After you've commited code changes and pushed your commits up to your fork, open your 
   fork on GitHub :code:`https://github.com/<your-gh-account>/espresso` in a browser.

#. Find the word "Contribute" on top of the page, click it and choose the green "Open 
   pull request" button. Follow the prompts and fill in necessary message you'd like us
   to know.

   ..    TODO insert a screenshot here

#. Once your pull request is submitted, some automatic checks will be triggered. Rest 
   assured - we will review your contribution, comment if necessary, and proceed to merge
   your contribution into our main repository when everything's ready.

   ..    TODO insert a screenshot here

#. Thanks again, for your contribution to open source 🌟 
