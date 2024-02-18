import React from 'react';

import config from '../config/index.json';

const Product = () => {
  const { product } = config;

  return (
    <section className={`bg-background py-8`} id="product">
      <div className="mx-auto px-4 ">
        <h1
          className={`w-full my-2 text-5xl font-bold leading-tight text-center text-primary`}
        >
          {product.title.split(' ').map((word, index) => (
            <span
              key={index}
              className={index % 2 ? 'text-primary' : 'text-border'}
            >
              {word}{' '}
            </span>
          ))}
        </h1>
        {/* <Divider /> */}
        <div className="lg:text-center">
          <p
            className={`mt-4 max-w-3xl text-2xl text-neutral-900 lg:mx-auto text-neutral-800`}
          >
            Convert image to Latex with incredible accuracy powered by AI
          </p>
        </div>

        {/* <div className={`flex flex-wrap flex-col-reverse sm:flex-row`}> */}
        {/*  <div className={`w-full sm:w-1/2 p-6`}> */}
        {/*    <img */}
        {/*      className="h-6/6" */}
        {/*      src={secondItem?.img} */}
        {/*      alt={secondItem?.title} */}
        {/*    /> */}
        {/*  </div> */}
        {/*  <div className={`w-full sm:w-1/2 p-6 mt-20`}> */}
        {/*    <div className={`align-middle`}> */}
        {/*      <h3 */}
        {/*        className={`text-3xl text-gray-800 font-bold leading-none mb-3`} */}
        {/*      > */}
        {/*        {secondItem?.title} */}
        {/*      </h3> */}
        {/*      <p className={`text-gray-600 mb-8`}>{secondItem?.description}</p> */}
        {/*    </div> */}
        {/*  </div> */}
        {/* </div> */}
      </div>
      {/* <div><ContactUs /></div> */}
    </section>
  );
};

export default Product;
