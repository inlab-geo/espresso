import subprocess
import pathlib
import shutil
import numpy as np
import matplotlib.pyplot as plt

from cofi_espresso import EspressoProblem
from cofi_espresso.exceptions import InvalidExampleError


LIB_DIR = pathlib.Path(__file__).resolve().parent / "lib"

class ReceiverFunction(EspressoProblem):
    """Forward simulation class
    """

    # TODO fill in the following metadata.
    metadata = {
        "problem_title": "Receiver function",                # To be used in docs
        "problem_short_description": "",    # 1-3 sentences

        "author_names": ["Malcolm Sambridge"],    # List of names e.g. author_names = ["Sally Smith", "Mark Brown"]

        "contact_name": "",         # Contact for contributor/maintainer of espresso example
        "contact_email": "",

        "citations": [("","")], # Reference to publication(s) that describe this example. In most 
                                # cases there will only be a single entry in this list.
                                # List of (citation, doi) pairs e.g. 
                                # citations = [("Newton, I (1687). Philosophiae naturalis principia mathematica.", "")]
                                # If there are no citations, use empty list `[]`

        "linked_sites": [("","")],  # List of (title, address) pairs for any websites that 
                                    # should be linked in the documentation, e.g.
                                    # linked_sites = [("Parent project on Github","https://github.com/user/repo"),
                                    #                 ("Data source"),"https://www.data.com") ]
                                    # If there are no links, use empty list `[]`
    }


    def __init__(self, example_number=1):
        super().__init__(example_number)

        """you might want to set some useful example-specific parameters here
        """

        try:
            from lib import rf
        except:
            build_clean()
            build_fortran_source()
            from lib import rf
        self.rf = rf

        if example_number == 1:
            self._good_model = np.array([[1,4.0,1.7],
                                    [3.5,4.3,1.7],
                                    [8.0,4.2,2.0],
                                    [20, 6,1.7],
                                    [45,6.2,1.7]])
        else:
            raise InvalidExampleError

    @property
    def description(self):
        raise NotImplementedError               # optional

    @property
    def model_size(self):
        raise NotImplementedError               # TODO implement me

    @property
    def data_size(self):
        raise NotImplementedError               # TODO implement me

    @property
    def good_model(self):
        return self._good_model

    @property
    def starting_model(self):
        raise NotImplementedError               # TODO implement me
    
    @property
    def data(self):
        raise NotImplementedError               # TODO implement me

    @property
    def covariance_matrix(self):                # optional
        raise NotImplementedError

    @property
    def inverse_covariance_matrix(self):
        raise NotImplementedError               # optional
        
    def forward(self, model, with_jacobian=False, *args, **kwargs):
        if with_jacobian:
            raise NotImplementedError           # optional
        else:
            t, rfunc = self.rf.rfcalc(model, *args, **kwargs)
            data_synth = np.vstack((t, rfunc)).T
            return data_synth
    
    def jacobian(self, model):
        raise NotImplementedError               # optional

    def plot_model(self, model):
        px = np.zeros([2*len(model),2])
        px[0::2,0],px[1::2,0],px[1::2,1],px[2::2,1] = model[:,1],model[:,1],model[:,0],model[:-1,0]
        fig, ax = plt.subplots(1, 1, figsize=(4,6))
        ax.set_xlabel('Vs (km/s)')
        ax.set_ylabel('Depth (km)')
        ax.invert_yaxis()
        ax.plot(px[:,0],px[:,1],'b-')
        return fig
    
    def plot_data(self, data, data2=None, label=None, label2=None):
        fig, ax = plt.subplots(1, 1)
        ax.plot(data[:,0],data[:,1],label=label)
        if data2 is not None:
            ax.plot(data2[:,0],data2[:,1],'r-',label=label2)
        ax.set_xlabel('Time/s')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend()
        return fig

    def misfit(self, data, data2):
        raise NotImplementedError               # optional

    def log_likelihood(self,data,data2):
        raise NotImplementedError               # optional
    
    def log_prior(self, model):
        raise NotImplementedError               # optional


def build_fortran_source():
    subprocess.run(["cmake", "-S", ".", "-B", "build"], cwd=LIB_DIR)
    subprocess.run(["cmake", "--build", "build"], cwd=LIB_DIR)
    subprocess.run(["cmake", "--build", "build"], cwd=LIB_DIR)

def build_clean():
    shutil.rmtree(LIB_DIR / "build", ignore_errors=True)


if __name__ == "__main__":
    try:
        from lib import rf
    except Exception as e:
        build_clean()
        build_fortran_source()

