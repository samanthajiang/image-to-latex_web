import React from 'react';

const Product = () => {
  return (
    <section className={`bg-background py-8`} id="product">
      <div className="mx-auto px-4 ">
        <h1
          className={`w-full my-2 text-5xl font-bold leading-tight text-center text-primary`}
        >
          {'Free Online Image to LaTeX Math Converter'
            .split(' ')
            .map((word, index) => (
              <span
                key={index}
                className={index % 2 ? 'text-primary' : 'text-border'}
              >
                {word}{' '}
              </span>
            ))}
        </h1>
        <div className="lg:text-center">
          <h2
            className={`mt-4 max-w-3xl text-2xl text-neutral-900 lg:mx-auto text-neutral-800`}
          >
            Convert image of Latex Math Equations to Editable Formula powered by
            AI
          </h2>
        </div>
      </div>
    </section>
  );
};

export default Product;
