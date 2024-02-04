import React from 'react';

import Canvas from '../components/Canvas';
import Contact from '../components/Email';
import Header from '../components/Header';
import Introduction from '../components/Intro';
import LazyShow from '../components/LazyShow';
import Product from '../components/Product';
import Upload_image from '../components/Upload_image';

const App = () => {
  return (
    <div className={`bg-background grid gap-y-16 overflow-hidden`}>
      <LazyShow>
        <>
          <Header />

          <Product />
          <Introduction />
          <Upload_image />

          {/* <Canvas /> */}
        </>
      </LazyShow>
      <LazyShow>
        <>
          <Contact />
          <Canvas />
        </>
      </LazyShow>
      {/* <LazyShow> */}
      {/*  <Pricing /> */}
      {/* </LazyShow> */}
      {/* <LazyShow> */}
      {/*  <> */}
      {/*    <Canvas /> */}
      {/*    <About /> */}
      {/*  </> */}
      {/* </LazyShow> */}
      {/* <Analytics /> */}
    </div>
  );
};

export default App;
