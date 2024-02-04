import React from 'react';

const Introduction = () => {
  return (
    <section className={`bg-red-50 ...`}>
      <div className="lg:text-center">
        <div className="mx-auto px-2 py-2 sm:px-6 lg:px-8 items-center">
          {/*<p*/}
          {/*  className={`mt-2 max-w-3xl text-2xl text-neutral-900 lg:mx-auto text-neutral-800`}*/}
          {/*>*/}
          {/*  Convert image to Latex with incredible accuracy powered by AI*/}

          {/*  </p> */}

{/*<></>flex flex-row justify-between items-stretch w-1/2 mx-auto*/}
          <div className=" container flex justify-around mx-auto items-start w-2/3">
            <div className=" w-1/2 flex flex-col justify-center items-center mx-auto">
              <p className={`mt-2 text-xl font-semibold`}>
                Simple & Complex Latex
              </p>

              <img className="w-3/4 px-2 pb-2 pt-4 " src="/assets/images/1.jpg" />
            </div>


          <div className=" w-1/2 flex flex-col justify-center items-center mx-auto">
            <p className={`mt-2 text-xl font-semibold`}>Predicted Formula </p>

          <img className=" p-2 " src="/assets/images/2.jpg" />
            </div>
            </div>
          <p className={`mt-2 max-w-3xl text-xl text-gray-500 lg:mx-auto text-gray-600`}>
            {/*Convert image to LaTeX formula for scientific documents like*/}
            {/*research papers*/}
            * Currently only support <span className={'text-primary'}>pure latex-formula</span> image


          </p>
        </div>
      </div>
    </section>
  );
};

export default Introduction;
